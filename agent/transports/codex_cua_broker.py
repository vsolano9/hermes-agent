"""Model-free broker for the signed Codex Computer Use MCP surface.

This module deliberately never starts an App Server turn. It creates one
ephemeral thread only to scope Codex's maintained MCP status/call methods.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import pwd
import re
import socket
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from agent.transports.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    UnixWebSocketCodexAppServerConnection,
)
from tools.mcp_pinned_surfaces import (
    CODEX_CUA_APP_SERVER_CATALOG_SHA256,
    CODEX_CUA_APP_SERVER_NAME,
    CODEX_CUA_APP_SERVER_PLUGIN_ID,
    CODEX_CUA_TOOL_NAMES,
    app_server_catalog_sha256,
)


CHATGPT_CODEX_PATH = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
CHATGPT_APP_PATH = Path("/Applications/ChatGPT.app")
OPENAI_TEAM_ID = "2DC432GLL2"
CHATGPT_BUNDLE_IDENTIFIER = "com.openai.codex"
CODEX_CLI_IDENTIFIER = "codex"
CODEX_CUA_SERVER_NAME = CODEX_CUA_APP_SERVER_NAME
CODEX_CUA_PLUGIN_ID = CODEX_CUA_APP_SERVER_PLUGIN_ID
APP_SERVER_SOCKET_NAME = "app-server-control.sock"
SUPPORTED_CODEX_APP_SERVER_VERSIONS = frozenset({"0.149.0-alpha.4.3"})
MAX_CATALOG_PAGES = 8
MAX_CATALOG_ROWS = 100
_VERSION_RE = re.compile(r"(?:codex-cli/|codex-cli\s+)?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)")
logger = logging.getLogger(__name__)


class CodexCUAError(RuntimeError):
    """Base class for fail-closed broker errors."""


class CodexCUABinaryUntrusted(CodexCUAError):
    pass


class CodexCUADaemonUnavailable(CodexCUAError):
    pass


class CodexCUAVersionMismatch(CodexCUAError):
    pass


class CodexCUACatalogDrift(CodexCUAError):
    pass


class CodexCUAProtocolError(CodexCUAError):
    pass


class CodexCUACallRejected(CodexCUAError):
    pass


class CodexCUAResultAmbiguous(CodexCUAError):
    pass


def _bounded_timeout(deadline: Optional[float], cap: float, phase: str) -> float:
    """Clamp one blocking operation to the caller's absolute deadline."""

    if deadline is None:
        return cap
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"Computer Use overall deadline expired during {phase}")
    return min(cap, remaining)


@dataclass(frozen=True)
class _PathSnapshot:
    path: str
    dev: int
    ino: int
    mode: int
    uid: int
    gid: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class VerifiedCodex:
    path: str
    version: str
    path_snapshot: tuple[_PathSnapshot, ...]

    def recheck(self) -> None:
        """Reject any path-component vnode or permission drift."""

        for expected in self.path_snapshot:
            try:
                current = os.lstat(expected.path)
            except OSError as exc:
                raise CodexCUABinaryUntrusted(
                    f"verified Codex path disappeared: {expected.path}"
                ) from exc
            actual = (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_uid,
                current.st_gid,
                current.st_nlink,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            wanted = (
                expected.dev,
                expected.ino,
                expected.mode,
                expected.uid,
                expected.gid,
                expected.nlink,
                expected.size,
                expected.mtime_ns,
                expected.ctime_ns,
            )
            if actual != wanted:
                raise CodexCUABinaryUntrusted(
                    f"verified Codex path changed before use: {expected.path}"
                )


@dataclass(frozen=True)
class ReadyDaemon:
    socket_path: str
    version: str
    verified: VerifiedCodex


def _account_home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)


def _minimal_daemon_env(codex_home: Path) -> dict[str, str]:
    """Return only the OS/runtime variables required by the local daemon."""

    env = {
        "HOME": str(_account_home()),
        "CODEX_HOME": str(codex_home),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "RUST_LOG": "warn",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", tempfile.gettempdir()),
    }
    return env


def _designated_requirement(identifier: str) -> str:
    return (
        f'identifier "{identifier}" and anchor apple generic and '
        "certificate 1[field.1.2.840.113635.100.6.2.6] exists and "
        "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and "
        f'certificate leaf[subject.OU] = "{OPENAI_TEAM_ID}"'
    )


def _codesign_identity(
    path: Path,
    expected_identifier: str,
    *,
    deadline: Optional[float] = None,
) -> tuple[str, str]:
    verify = subprocess.run(
        [
            "/usr/bin/codesign", "--verify", "--deep", "--strict",
            "--verbose=2", "-R", "=" + _designated_requirement(expected_identifier),
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_bounded_timeout(deadline, 15.0, "codesign verification"),
        stdin=subprocess.DEVNULL,
    )
    if verify.returncode != 0:
        raise CodexCUABinaryUntrusted(f"codesign verification failed for {path}")
    details = subprocess.run(
        ["/usr/bin/codesign", "-dv", "--verbose=4", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_bounded_timeout(deadline, 15.0, "codesign identity inspection"),
        stdin=subprocess.DEVNULL,
    )
    output = details.stdout + "\n" + details.stderr
    identifier = re.search(r"^Identifier=(.+)$", output, re.MULTILINE)
    team = re.search(r"^TeamIdentifier=(.+)$", output, re.MULTILINE)
    if details.returncode != 0 or identifier is None or team is None:
        raise CodexCUABinaryUntrusted(f"could not attest codesign identity for {path}")
    return identifier.group(1).strip(), team.group(1).strip()


def _trusted_path_snapshot(path: Path) -> tuple[_PathSnapshot, ...]:
    """Validate and snapshot every component of the fixed application path."""

    if not path.is_absolute():
        raise CodexCUABinaryUntrusted("Codex path must be absolute")
    uid = os.getuid()
    components = [Path("/")]
    cursor = Path("/")
    for part in path.parts[1:]:
        cursor = cursor / part
        components.append(cursor)
    snapshots = []
    for component in components:
        try:
            info = os.lstat(component)
        except OSError as exc:
            raise CodexCUABinaryUntrusted(
                f"Codex path component is unavailable: {component}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise CodexCUABinaryUntrusted(
                f"Codex path component is a symlink: {component}"
            )
        if info.st_uid not in {0, uid}:
            raise CodexCUABinaryUntrusted(
                f"Codex path component has an unexpected owner: {component}"
            )
        # /Applications is root-owned and group-writable by the macOS admin
        # group on a stock installation. All application-owned descendants
        # must be non-writable by group and world.
        if component not in {Path("/"), Path("/Applications")} and (
            info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise CodexCUABinaryUntrusted(
                f"Codex path component is writable by other principals: {component}"
            )
        snapshots.append(_PathSnapshot(
            str(component), info.st_dev, info.st_ino, info.st_mode,
            info.st_uid, info.st_gid, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns,
        ))
    if not stat.S_ISREG(snapshots[-1].mode) or not os.access(path, os.X_OK):
        raise CodexCUABinaryUntrusted("signed Codex CLI is not an executable file")
    return tuple(snapshots)


def resolve_verified_codex(
    path: Path = CHATGPT_CODEX_PATH,
    *,
    deadline: Optional[float] = None,
) -> VerifiedCodex:
    """Resolve only the signed Codex binary embedded in ChatGPT.app."""

    if path != CHATGPT_CODEX_PATH:
        raise CodexCUABinaryUntrusted("only the ChatGPT-embedded Codex CLI is supported")
    snapshot = _trusted_path_snapshot(path)
    bundle_id, bundle_team = _codesign_identity(
        CHATGPT_APP_PATH, CHATGPT_BUNDLE_IDENTIFIER, deadline=deadline
    )
    cli_id, cli_team = _codesign_identity(
        path, CODEX_CLI_IDENTIFIER, deadline=deadline
    )
    if (bundle_id, bundle_team) != (CHATGPT_BUNDLE_IDENTIFIER, OPENAI_TEAM_ID):
        raise CodexCUABinaryUntrusted("ChatGPT Codex bundle identity is not trusted")
    if (cli_id, cli_team) != (CODEX_CLI_IDENTIFIER, OPENAI_TEAM_ID):
        raise CodexCUABinaryUntrusted("embedded Codex CLI identity is not trusted")
    proc = subprocess.run(
        [str(path), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_bounded_timeout(deadline, 10.0, "Codex version inspection"),
        stdin=subprocess.DEVNULL,
        env=_minimal_daemon_env(_account_home() / ".codex"),
    )
    match = _VERSION_RE.search(proc.stdout or "")
    if proc.returncode != 0 or match is None:
        raise CodexCUABinaryUntrusted("embedded Codex CLI version is unavailable")
    verified = VerifiedCodex(str(path), match.group(1), snapshot)
    verified.recheck()
    return verified


def _snapshot_path(path: Path) -> _PathSnapshot:
    info = os.lstat(path)
    return _PathSnapshot(
        str(path), info.st_dev, info.st_ino, info.st_mode, info.st_uid,
        info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _resolve_verified_managed_codex(
    reported_path: str,
    version: str,
    codex_home: Path,
    *,
    deadline: Optional[float] = None,
) -> VerifiedCodex:
    """Bind daemon JSON to the signed standalone release that owns the UDS."""

    expected_alias = codex_home / "packages" / "standalone" / "current" / "codex"
    if reported_path != str(expected_alias):
        raise CodexCUABinaryUntrusted("managed Codex path is not the official alias")
    machine = os.uname().machine
    platform = {"arm64": "aarch64", "x86_64": "x86_64"}.get(machine)
    if platform is None:
        raise CodexCUABinaryUntrusted("managed Codex platform is unsupported")
    release_root = codex_home / "packages" / "standalone" / "releases"
    release_dir = release_root / f"{version}-{platform}-apple-darwin"
    expected_path = release_dir / "bin" / "codex"
    try:
        actual_path = expected_alias.resolve(strict=True)
    except OSError as exc:
        raise CodexCUABinaryUntrusted("managed Codex path is unavailable") from exc
    if actual_path != expected_path:
        raise CodexCUABinaryUntrusted("managed Codex path/version layout drifted")

    current_link = expected_alias.parent
    package_link = release_dir / "codex"
    try:
        current_info = os.lstat(current_link)
        package_info = os.lstat(package_link)
    except OSError as exc:
        raise CodexCUABinaryUntrusted("managed Codex alias chain is unavailable") from exc
    if (
        not stat.S_ISLNK(current_info.st_mode)
        or current_info.st_uid != os.getuid()
        or current_link.resolve(strict=True) != release_dir
        or not stat.S_ISLNK(package_info.st_mode)
        or package_info.st_uid != os.getuid()
        or os.readlink(package_link) != "bin/codex"
    ):
        raise CodexCUABinaryUntrusted("managed Codex alias chain drifted")

    snapshot = _trusted_path_snapshot(actual_path) + (
        _snapshot_path(current_link),
        _snapshot_path(package_link),
    )
    cli_id, cli_team = _codesign_identity(
        actual_path, CODEX_CLI_IDENTIFIER, deadline=deadline
    )
    if (cli_id, cli_team) != (CODEX_CLI_IDENTIFIER, OPENAI_TEAM_ID):
        raise CodexCUABinaryUntrusted("managed Codex CLI identity is not trusted")
    inspected = subprocess.run(
        [str(actual_path), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_bounded_timeout(deadline, 10.0, "managed Codex version inspection"),
        stdin=subprocess.DEVNULL,
        env=_minimal_daemon_env(codex_home),
    )
    match = _VERSION_RE.search(inspected.stdout or "")
    if inspected.returncode != 0 or match is None:
        raise CodexCUABinaryUntrusted("managed Codex CLI version is unavailable")
    if match.group(1) != version:
        raise CodexCUAVersionMismatch("managed Codex binary version drifted")
    verified = VerifiedCodex(str(actual_path), version, snapshot)
    verified.recheck()
    return verified


def _validate_private_directory(path: Path, *, create: bool) -> None:
    if create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    info = os.lstat(path)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise CodexCUADaemonUnavailable(
            "Codex app-server-control directory is not private and account-owned"
        )


def _validate_codex_home(path: Path) -> None:
    """Validate the existing account-owned state root without requiring 0700."""

    info = os.lstat(path)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise CodexCUADaemonUnavailable(
            "CODEX_HOME is linked, writable by others, or not account-owned"
        )


def _validate_private_socket(path: Path) -> None:
    info = os.lstat(path)
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISSOCK(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & (stat.S_IRWXG | stat.S_IRWXO)
    ):
        raise CodexCUADaemonUnavailable(
            "Codex app-server socket is not private and account-owned"
        )


class _CFDictionaryKeyCallBacks(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_long),
        ("retain", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("copy_description", ctypes.c_void_p),
        ("equal", ctypes.c_void_p),
        ("hash", ctypes.c_void_p),
    ]


class _CFDictionaryValueCallBacks(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_long),
        ("retain", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("copy_description", ctypes.c_void_p),
        ("equal", ctypes.c_void_p),
    ]


def _peer_executable_path(pid: int) -> str:
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        proc_pidpath = libproc.proc_pidpath
        proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        proc_pidpath.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(4096)
        length = proc_pidpath(pid, buffer, len(buffer))
    except (AttributeError, OSError) as exc:
        raise CodexCUABinaryUntrusted("could not inspect App Server peer path") from exc
    if length <= 0:
        raise CodexCUABinaryUntrusted("could not inspect App Server peer path")
    return buffer.value.decode("utf-8", "strict")


def _validate_peer_designated_requirement(pid: int) -> None:
    """Bind a live PID to the exact signed OpenAI CLI requirement."""

    try:
        core = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        security = ctypes.CDLL(
            "/System/Library/Frameworks/Security.framework/Security"
        )
        key_callbacks = _CFDictionaryKeyCallBacks.in_dll(
            core, "kCFTypeDictionaryKeyCallBacks"
        )
        value_callbacks = _CFDictionaryValueCallBacks.in_dll(
            core, "kCFTypeDictionaryValueCallBacks"
        )
        pid_key = ctypes.c_void_p.in_dll(
            security, "kSecGuestAttributePid"
        ).value
    except (OSError, ValueError) as exc:
        raise CodexCUABinaryUntrusted(
            "macOS code-signing services are unavailable for App Server peer"
        ) from exc

    core.CFRelease.argtypes = [ctypes.c_void_p]
    core.CFNumberCreate.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
    core.CFNumberCreate.restype = ctypes.c_void_p
    core.CFDictionaryCreate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_long,
        ctypes.POINTER(_CFDictionaryKeyCallBacks),
        ctypes.POINTER(_CFDictionaryValueCallBacks),
    ]
    core.CFDictionaryCreate.restype = ctypes.c_void_p
    core.CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32
    ]
    core.CFStringCreateWithCString.restype = ctypes.c_void_p
    security.SecCodeCopyGuestWithAttributes.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    security.SecCodeCopyGuestWithAttributes.restype = ctypes.c_int32
    security.SecRequirementCreateWithString.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)
    ]
    security.SecRequirementCreateWithString.restype = ctypes.c_int32
    security.SecCodeCheckValidity.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p
    ]
    security.SecCodeCheckValidity.restype = ctypes.c_int32

    number = dictionary = requirement_string = requirement = code = None
    try:
        pid_value = ctypes.c_int32(pid)
        number = core.CFNumberCreate(None, 3, ctypes.byref(pid_value))
        if not number or not pid_key:
            raise CodexCUABinaryUntrusted("could not bind App Server peer PID")
        keys = (ctypes.c_void_p * 1)(pid_key)
        values = (ctypes.c_void_p * 1)(number)
        dictionary = core.CFDictionaryCreate(
            None, keys, values, 1,
            ctypes.byref(key_callbacks), ctypes.byref(value_callbacks),
        )
        requirement_string = core.CFStringCreateWithCString(
            None,
            _designated_requirement(CODEX_CLI_IDENTIFIER).encode("utf-8"),
            0x08000100,  # kCFStringEncodingUTF8
        )
        if not dictionary or not requirement_string:
            raise CodexCUABinaryUntrusted("could not construct peer attestation")
        requirement_ref = ctypes.c_void_p()
        if security.SecRequirementCreateWithString(
            requirement_string, 0, ctypes.byref(requirement_ref)
        ) != 0:
            raise CodexCUABinaryUntrusted("could not construct peer requirement")
        requirement = requirement_ref.value
        code_ref = ctypes.c_void_p()
        if security.SecCodeCopyGuestWithAttributes(
            None, dictionary, 0, ctypes.byref(code_ref)
        ) != 0:
            raise CodexCUABinaryUntrusted("App Server peer is not valid signed code")
        code = code_ref.value
        if security.SecCodeCheckValidity(code, 0, requirement) != 0:
            raise CodexCUABinaryUntrusted(
                "App Server peer does not satisfy the OpenAI CLI requirement"
            )
    finally:
        for reference in (code, requirement, requirement_string, dictionary, number):
            if reference:
                core.CFRelease(reference)


def _validate_codex_peer(peer_socket: socket.socket, verified: VerifiedCodex) -> None:
    """Authenticate the already-connected Unix peer before WebSocket upgrade."""

    try:
        pid = peer_socket.getsockopt(0, 0x002)  # SOL_LOCAL / LOCAL_PEERPID
    except OSError as exc:
        raise CodexCUABinaryUntrusted("App Server peer PID is unavailable") from exc
    if not isinstance(pid, int) or pid <= 0:
        raise CodexCUABinaryUntrusted("App Server peer PID is invalid")
    if _peer_executable_path(pid) != verified.path:
        raise CodexCUABinaryUntrusted("App Server peer executable is not verified Codex")
    _validate_peer_designated_requirement(pid)
    verified.recheck()


class CodexAppServerDaemonController:
    """Start or reuse Codex's account-global managed local daemon."""

    def __init__(self, codex_home: Optional[Path] = None) -> None:
        home = codex_home or (_account_home() / ".codex")
        self.codex_home = Path(home).absolute()

    def ensure_ready(
        self,
        verified: VerifiedCodex,
        *,
        deadline: Optional[float] = None,
    ) -> ReadyDaemon:
        control = self.codex_home / "app-server-control"
        _validate_codex_home(self.codex_home)
        _validate_private_directory(control, create=True)
        verified.recheck()
        env = _minimal_daemon_env(self.codex_home)
        started = subprocess.run(
            [verified.path, "app-server", "daemon", "start"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_bounded_timeout(deadline, 30.0, "daemon start"),
            stdin=subprocess.DEVNULL,
            env=env,
        )
        if started.returncode != 0:
            raise CodexCUADaemonUnavailable("signed Codex app-server daemon did not start")
        verified.recheck()
        version_result = subprocess.run(
            [verified.path, "app-server", "daemon", "version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_bounded_timeout(deadline, 10.0, "daemon version inspection"),
            stdin=subprocess.DEVNULL,
            env=env,
        )
        try:
            versions = json.loads(version_result.stdout)
        except (TypeError, ValueError) as exc:
            raise CodexCUADaemonUnavailable("daemon version response was invalid") from exc
        if not isinstance(versions, dict):
            raise CodexCUADaemonUnavailable("daemon version response was invalid")
        managed_path = versions.get("managedCodexPath")
        managed_version = versions.get("managedCodexVersion")
        daemon_version = versions.get("appServerVersion")
        launcher_version = versions.get("cliVersion")
        reported_socket = versions.get("socketPath")
        if (
            version_result.returncode != 0
            or versions.get("status") != "running"
            or versions.get("backend") != "pid"
            or not isinstance(managed_path, str)
            or not isinstance(managed_version, str)
            or not isinstance(daemon_version, str)
            or not isinstance(launcher_version, str)
            or not isinstance(reported_socket, str)
        ):
            raise CodexCUADaemonUnavailable("running daemon version is unavailable")
        socket_path = control / APP_SERVER_SOCKET_NAME
        if launcher_version != verified.version:
            raise CodexCUAVersionMismatch("daemon controller CLI version drifted")
        if managed_version != daemon_version:
            raise CodexCUAVersionMismatch("managed Codex and App Server versions differ")
        if daemon_version not in SUPPORTED_CODEX_APP_SERVER_VERSIONS:
            raise CodexCUAVersionMismatch("App Server protocol version is unsupported")
        if reported_socket != str(socket_path):
            raise CodexCUADaemonUnavailable("daemon socket path drifted")
        managed = _resolve_verified_managed_codex(
            managed_path, managed_version, self.codex_home, deadline=deadline
        )
        managed.recheck()
        verified.recheck()
        socket_deadline = time.monotonic() + 5.0
        if deadline is not None:
            socket_deadline = min(socket_deadline, deadline)
        while True:
            try:
                _validate_private_socket(socket_path)
                break
            except FileNotFoundError:
                if time.monotonic() >= socket_deadline:
                    raise CodexCUADaemonUnavailable("App Server socket did not appear")
                time.sleep(min(0.05, max(0.0, socket_deadline - time.monotonic())))
        managed.recheck()
        return ReadyDaemon(str(socket_path), daemon_version, managed)


class CodexCUABroker:
    """Perform one exact Computer Use call without invoking a Codex model."""

    def __init__(
        self,
        *,
        binary_resolver: Callable[..., VerifiedCodex] = resolve_verified_codex,
        daemon_controller: Optional[CodexAppServerDaemonController] = None,
        client_factory: Optional[Callable[[str, float], CodexAppServerClient]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self._binary_resolver = binary_resolver
        self._daemon = daemon_controller or CodexAppServerDaemonController()
        self._client_factory = client_factory
        self._cwd = cwd or str(_account_home())

    @staticmethod
    def _make_client(
        socket_path: str, verified: VerifiedCodex, open_timeout: float
    ) -> CodexAppServerClient:
        return CodexAppServerClient(
            connection=UnixWebSocketCodexAppServerConnection(
                socket_path,
                open_timeout=open_timeout,
                peer_validator=lambda peer: _validate_codex_peer(peer, verified),
            ),
            reject_server_requests=True,
            event_queue_limit=4,
        )

    def call(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        timeout: float,
    ) -> dict:
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise CodexCUAProtocolError("Computer Use timeout must be positive and finite")
        deadline = time.monotonic() + timeout
        if tool not in CODEX_CUA_TOOL_NAMES:
            raise CodexCUAProtocolError(f"unpublished Computer Use tool: {tool}")
        if not isinstance(arguments, Mapping):
            raise CodexCUAProtocolError("Computer Use arguments must be a mapping")
        launcher = self._binary_resolver(deadline=deadline)
        ready = self._daemon.ensure_ready(launcher, deadline=deadline)
        if (
            ready.version not in SUPPORTED_CODEX_APP_SERVER_VERSIONS
            or ready.version != ready.verified.version
        ):
            raise CodexCUAVersionMismatch("managed App Server version is unsupported")
        ready.verified.recheck()
        self._remaining(deadline, "daemon readiness")
        open_timeout = self._remaining(deadline, "socket connection")
        client = (
            self._client_factory(ready.socket_path, open_timeout)
            if self._client_factory is not None
            else self._make_client(ready.socket_path, ready.verified, open_timeout)
        )
        thread_id: Optional[str] = None
        cleanup_error: Optional[BaseException] = None
        try:
            self._remaining(deadline, "socket connection")
            initialized = client.initialize(
                timeout=min(self._remaining(deadline, "initialize"), 10.0)
            )
            if not isinstance(initialized, dict):
                raise CodexCUAProtocolError("App Server initialize result is invalid")
            agent_version = _VERSION_RE.search(str(initialized.get("userAgent") or ""))
            if agent_version is None or agent_version.group(1) != ready.version:
                raise CodexCUAVersionMismatch("connected App Server version differs")
            started = client.request(
                "thread/start",
                {"cwd": self._cwd, "ephemeral": True},
                timeout=self._remaining(deadline, "thread/start"),
            )
            if not isinstance(started, dict):
                raise CodexCUAProtocolError("thread/start result is invalid")
            thread = started.get("thread")
            if not isinstance(thread, dict):
                raise CodexCUAProtocolError("thread/start omitted the thread")
            thread_id = thread.get("id")
            if (
                not isinstance(thread_id, str)
                or not thread_id
                or thread.get("ephemeral") is not True
                or thread.get("path") is not None
            ):
                raise CodexCUAProtocolError("App Server did not create an ephemeral thread")
            self._attest_catalog(client, thread_id, deadline)
            try:
                result = client.request(
                    "mcpServer/tool/call",
                    {
                        "server": CODEX_CUA_SERVER_NAME,
                        "threadId": thread_id,
                        "tool": tool,
                        "arguments": dict(arguments),
                    },
                    timeout=self._remaining(deadline, "tool call"),
                )
                if not isinstance(result, dict):
                    raise CodexCUAProtocolError(
                        "App Server Computer Use result shape is invalid"
                    )
                try:
                    # App Server bypasses the MCP SDK transport, so validate
                    # against the very same SDK union used by ordinary MCP
                    # calls before sanitation/rendering. Strict validation
                    # prevents coercion of malformed post-dispatch results.
                    from mcp.types import CallToolResult

                    validated = CallToolResult.model_validate(result, strict=True)
                except (TypeError, ValueError) as exc:
                    raise CodexCUAProtocolError(
                        "App Server Computer Use result shape is invalid"
                    ) from exc
                normalized_result = validated.model_dump(
                    mode="python", by_alias=True, exclude_none=False
                )
                try:
                    json.dumps(normalized_result, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise CodexCUAProtocolError(
                        "App Server Computer Use result contains non-JSON values"
                    ) from exc
                return normalized_result
            except CodexAppServerError as exc:
                if exc.code == -10000:
                    raise CodexCUACallRejected(
                        "signed Computer Use service rejected App Server authentication"
                    ) from exc
                raise CodexCUAResultAmbiguous(
                    "Computer Use result is unknown after an accepted request frame; "
                    "do not retry this action"
                ) from exc
            except CodexCUAProtocolError as exc:
                raise CodexCUAResultAmbiguous(
                    "Computer Use returned an invalid result after an accepted request "
                    "frame; do not retry this action"
                ) from exc
            except Exception as exc:
                # request() only returns after `_send` accepted the complete
                # frame. Losing the response after that point is ambiguous and
                # must never be replayed. Preserve BaseException cancellation
                # semantics, but never expose an ordinary unexpected exception
                # as a retryable-looking tool failure.
                raise CodexCUAResultAmbiguous(
                    "Computer Use result is unknown after an accepted request frame; "
                    "do not retry this action"
                ) from exc
        finally:
            if thread_id is not None:
                try:
                    client.request(
                        "thread/delete", {"threadId": thread_id}, timeout=min(timeout, 5.0)
                    )
                except BaseException as exc:  # record without masking call truth
                    cleanup_error = exc
            try:
                client.close()
            except Exception as exc:
                # A known tool result plus successful thread/delete must not
                # become an apparent failure that encourages replay.
                logger.warning("Codex App Server connection close failed: %s", exc)
            if cleanup_error is not None:
                # Never hide a known result or primary exception: presenting
                # cleanup loss as the call result can induce duplicate writes.
                logger.warning(
                    "Ephemeral Codex App Server thread deletion failed: %s",
                    cleanup_error,
                )

    @staticmethod
    def _remaining(deadline: float, phase: str) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Computer Use overall deadline expired during {phase}")
        return remaining

    @staticmethod
    def _attest_catalog(client: CodexAppServerClient, thread_id: str, deadline: float) -> None:
        rows = []
        cursor = None
        seen_cursors: set[str] = set()
        for _page_number in range(MAX_CATALOG_PAGES):
            try:
                remaining = CodexCUABroker._remaining(deadline, "catalog attestation")
            except TimeoutError as exc:
                raise CodexCUACatalogDrift(
                    "Computer Use catalog attestation exceeded the overall deadline"
                ) from exc
            params = {
                "threadId": thread_id,
                "cursor": cursor,
                "limit": 100,
                "detail": "full",
            }
            page = client.request("mcpServerStatus/list", params, timeout=remaining)
            if not isinstance(page, dict):
                raise CodexCUACatalogDrift("App Server MCP catalog page is invalid")
            data = page.get("data")
            if not isinstance(data, list):
                raise CodexCUACatalogDrift("App Server MCP catalog data is invalid")
            if len(data) > MAX_CATALOG_ROWS or len(rows) + len(data) > MAX_CATALOG_ROWS:
                raise CodexCUACatalogDrift("App Server MCP catalog item limit exceeded")
            rows.extend(data)
            cursor = page.get("nextCursor")
            if cursor is None:
                break
            if not isinstance(cursor, str) or not cursor:
                raise CodexCUACatalogDrift("App Server MCP catalog cursor is invalid")
            if cursor in seen_cursors:
                raise CodexCUACatalogDrift("App Server MCP catalog cursor repeated")
            seen_cursors.add(cursor)
        else:
            raise CodexCUACatalogDrift("App Server MCP catalog page limit exceeded")
        matches = [
            row for row in rows
            if isinstance(row, dict) and (
                row.get("name") == CODEX_CUA_SERVER_NAME
                or row.get("pluginId") == CODEX_CUA_PLUGIN_ID
            )
        ]
        if len(matches) != 1:
            raise CodexCUACatalogDrift("Computer Use MCP row is missing or duplicated")
        row = matches[0]
        try:
            digest = app_server_catalog_sha256(row)
        except (TypeError, ValueError) as exc:
            raise CodexCUACatalogDrift(
                "Computer Use MCP catalog attestation drifted"
            ) from exc
        if digest != CODEX_CUA_APP_SERVER_CATALOG_SHA256:
            raise CodexCUACatalogDrift("Computer Use MCP catalog attestation drifted")
