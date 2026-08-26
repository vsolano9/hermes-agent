"""Trust-boundary tests for the replacement Codex App Server broker."""

from __future__ import annotations

import ctypes
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace

import pytest

from agent.transports import codex_cua_broker as broker


def _executable(tmp_path: Path) -> Path:
    path = tmp_path / "trusted" / "codex"
    path.parent.mkdir()
    path.write_bytes(b"fixture")
    path.chmod(0o755)
    return path


def test_path_snapshot_rejects_symlinked_component(tmp_path) -> None:
    executable = _executable(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(executable.parent, target_is_directory=True)

    real_lstat = broker.os.lstat

    def safe_ancestors(path):
        info = real_lstat(path)
        if Path(path) in {linked}:
            return info
        # Test temp roots are intentionally world-writable. Mask only their
        # write bits so this case reaches the symlink under test.
        values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        values["st_mode"] = info.st_mode & ~0o022
        return type("Stat", (), values)()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(broker.os, "lstat", safe_ancestors)
        with pytest.raises(broker.CodexCUABinaryUntrusted, match="symlink"):
            broker._trusted_path_snapshot(linked / "codex")


def test_path_snapshot_rejects_group_writable_application_component(tmp_path) -> None:
    executable = _executable(tmp_path)
    executable.parent.chmod(0o775)

    with pytest.raises(broker.CodexCUABinaryUntrusted, match="writable"):
        broker._trusted_path_snapshot(executable)


def test_verified_snapshot_detects_executable_replacement(tmp_path) -> None:
    executable = _executable(tmp_path)
    info = executable.lstat()
    snapshot = (broker._PathSnapshot(
        str(executable), info.st_dev, info.st_ino, info.st_mode, info.st_uid,
        info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns,
        info.st_ctime_ns,
    ),)
    verified = broker.VerifiedCodex(str(executable), "0.149.0", snapshot)
    executable.unlink()
    executable.write_bytes(b"replacement")
    executable.chmod(0o755)

    with pytest.raises(broker.CodexCUABinaryUntrusted, match="changed"):
        verified.recheck()


def test_verified_snapshot_detects_same_size_write_with_restored_mtime(tmp_path) -> None:
    executable = _executable(tmp_path)
    info = executable.lstat()
    snapshot = (broker._PathSnapshot(
        str(executable), info.st_dev, info.st_ino, info.st_mode, info.st_uid,
        info.st_gid, info.st_nlink, info.st_size, info.st_mtime_ns,
        info.st_ctime_ns,
    ),)
    verified = broker.VerifiedCodex(str(executable), "0.149.0", snapshot)
    executable.write_bytes(b"changed")  # same size as the original fixture
    broker.os.utime(executable, ns=(info.st_atime_ns, info.st_mtime_ns))

    with pytest.raises(broker.CodexCUABinaryUntrusted, match="changed"):
        verified.recheck()


def test_daemon_environment_drops_unrelated_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(broker, "_account_home", lambda: tmp_path)
    monkeypatch.setenv("SECRET_CANARY", "must-not-propagate")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-propagate")

    env = broker._minimal_daemon_env(tmp_path / ".codex")

    assert env["HOME"] == str(tmp_path)
    assert env["CODEX_HOME"] == str(tmp_path / ".codex")
    assert env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert "SECRET_CANARY" not in env
    assert "OPENAI_API_KEY" not in env


def test_codesign_verification_enforces_exact_apple_designated_requirement(
    monkeypatch,
) -> None:
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if "-dv" in argv:
            return SimpleNamespace(
                returncode=0,
                stdout="",
                stderr=(
                    "Identifier=com.openai.codex\n"
                    "TeamIdentifier=2DC432GLL2\n"
                ),
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(broker.subprocess, "run", fake_run)
    assert broker._codesign_identity(
        broker.CHATGPT_APP_PATH, broker.CHATGPT_BUNDLE_IDENTIFIER
    ) == (
        "com.openai.codex",
        "2DC432GLL2",
    )

    requirement = calls[0][calls[0].index("-R") + 1]
    assert requirement == (
        '=identifier "com.openai.codex" and anchor apple generic and '
        "certificate 1[field.1.2.840.113635.100.6.2.6] exists and "
        "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and "
        'certificate leaf[subject.OU] = "2DC432GLL2"'
    )


def test_security_framework_requirement_omits_codesign_cli_prefix() -> None:
    """Security.framework rejects codesign's leading ``=`` wrapper."""

    requirement = broker._designated_requirement(broker.CODEX_CLI_IDENTIFIER)

    assert requirement == (
        'identifier "codex" and anchor apple generic and '
        "certificate 1[field.1.2.840.113635.100.6.2.6] exists and "
        "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and "
        'certificate leaf[subject.OU] = "2DC432GLL2"'
    )


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="Security.framework requirement parsing is macOS-only",
)
def test_security_framework_parses_exact_peer_requirement_before_validity() -> None:
    """The no-prefix requirement parses; identity failure is a later check."""

    core = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    security = ctypes.CDLL(
        "/System/Library/Frameworks/Security.framework/Security"
    )
    core.CFRelease.argtypes = [ctypes.c_void_p]
    core.CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    core.CFStringCreateWithCString.restype = ctypes.c_void_p
    security.SecRequirementCreateWithString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    security.SecRequirementCreateWithString.restype = ctypes.c_int32
    security.SecCodeCopySelf.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    security.SecCodeCopySelf.restype = ctypes.c_int32
    security.SecCodeCheckValidity.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    security.SecCodeCheckValidity.restype = ctypes.c_int32

    requirement_text = broker._designated_requirement(broker.CODEX_CLI_IDENTIFIER)
    assert not requirement_text.startswith("=")
    requirement_string = core.CFStringCreateWithCString(
        None,
        requirement_text.encode("utf-8"),
        0x08000100,  # kCFStringEncodingUTF8
    )
    requirement_ref = ctypes.c_void_p()
    self_code_ref = ctypes.c_void_p()
    try:
        assert requirement_string
        parse_status = security.SecRequirementCreateWithString(
            requirement_string,
            0,
            ctypes.byref(requirement_ref),
        )
        assert parse_status == 0
        assert requirement_ref.value

        # Parsing and trust evaluation are distinct. The pytest interpreter is
        # valid code, but it must not satisfy the exact OpenAI Codex identity.
        assert security.SecCodeCopySelf(0, ctypes.byref(self_code_ref)) == 0
        assert self_code_ref.value
        assert (
            security.SecCodeCheckValidity(
                self_code_ref,
                0,
                requirement_ref,
            )
            != 0
        )
    finally:
        for reference in (
            self_code_ref.value,
            requirement_ref.value,
            requirement_string,
        ):
            if reference:
                core.CFRelease(reference)


def test_private_control_directory_rejects_loose_mode(tmp_path) -> None:
    control = tmp_path / "app-server-control"
    control.mkdir(mode=0o755)

    with pytest.raises(broker.CodexCUADaemonUnavailable, match="not private"):
        broker._validate_private_directory(control, create=False)


def test_private_socket_rejects_regular_file(tmp_path) -> None:
    endpoint = tmp_path / "app-server-control.sock"
    endpoint.write_bytes(b"not a socket")
    endpoint.chmod(0o600)

    with pytest.raises(broker.CodexCUADaemonUnavailable, match="socket"):
        broker._validate_private_socket(endpoint)


def test_private_socket_rejects_wrong_owner(monkeypatch, tmp_path) -> None:
    endpoint = tmp_path / "app-server-control.sock"
    endpoint.write_bytes(b"fixture")
    real_lstat = broker.os.lstat

    def wrong_owner(path):
        info = real_lstat(path)
        values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
        values["st_mode"] = broker.stat.S_IFSOCK | 0o600
        values["st_uid"] = broker.os.getuid() + 1
        return type("Stat", (), values)()

    monkeypatch.setattr(broker.os, "lstat", wrong_owner)
    with pytest.raises(broker.CodexCUADaemonUnavailable, match="account-owned"):
        broker._validate_private_socket(endpoint)


def test_only_fixed_chatgpt_embedded_codex_path_can_be_resolved(tmp_path) -> None:
    with pytest.raises(broker.CodexCUABinaryUntrusted, match="only the"):
        broker.resolve_verified_codex(tmp_path / "codex")


@pytest.mark.parametrize(
    "version_payload,error_type",
    [
        (
            '{"managedCodexVersion":"0.149.0-alpha.4.3",'
            '"appServerVersion":"0.150.0"}',
            broker.CodexCUAVersionMismatch,
        ),
        ("[]", broker.CodexCUADaemonUnavailable),
    ],
)
def test_daemon_controller_fails_closed_on_version_response_drift(
    monkeypatch, tmp_path, version_payload, error_type
) -> None:
    codex_home = tmp_path / "codex-home"
    control = codex_home / "app-server-control"
    control.mkdir(parents=True, mode=0o700)
    if version_payload != "[]":
        version_payload = (
            '{"status":"running","backend":"pid",'
            f'"managedCodexPath":"{codex_home}/packages/standalone/current/codex",'
            '"managedCodexVersion":"0.149.0-alpha.4.3",'
            f'"socketPath":"{control / broker.APP_SERVER_SOCKET_NAME}",'
            '"cliVersion":"0.149.0-alpha.4.3",'
            '"appServerVersion":"0.150.0"}'
        )

    def fake_run(argv, **kwargs):
        if argv[-1] == "version":
            return SimpleNamespace(
                returncode=0, stdout=version_payload, stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(broker.subprocess, "run", fake_run)
    verified = broker.VerifiedCodex(
        "/signed/codex", "0.149.0-alpha.4.3", ()
    )
    with pytest.raises(error_type):
        broker.CodexAppServerDaemonController(codex_home).ensure_ready(verified)


def test_daemon_controller_uses_idempotent_managed_commands_and_private_socket(
    monkeypatch,
) -> None:
    # macOS limits AF_UNIX names to 104 bytes; pytest's nested tmp_path can
    # exceed that even though the production CODEX_HOME path does not.
    with tempfile.TemporaryDirectory(prefix="hcua-", dir="/tmp") as root:
        codex_home = Path(root) / "codex-home"
        control = codex_home / "app-server-control"
        control.mkdir(parents=True, mode=0o700)
        socket_path = control / broker.APP_SERVER_SOCKET_NAME
        endpoint = socket.socket(socket.AF_UNIX)
        endpoint.bind(str(socket_path))
        socket_path.chmod(0o600)
        calls = []

        def fake_run(argv, **kwargs):
            calls.append((list(argv), dict(kwargs)))
            if argv[-1] == "version":
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        '{"status":"running","backend":"pid",'
                        f'"managedCodexPath":"{codex_home}/packages/standalone/current/codex",'
                        '"managedCodexVersion":"0.149.0-alpha.4.3",'
                        f'"socketPath":"{socket_path}",'
                        '"cliVersion":"0.149.0-alpha.4.3",'
                        '"appServerVersion":"0.149.0-alpha.4.3"}'
                    ),
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")

        monkeypatch.setattr(broker.subprocess, "run", fake_run)
        managed = broker.VerifiedCodex(
            "/managed/codex", "0.149.0-alpha.4.3", ()
        )
        monkeypatch.setattr(
            broker,
            "_resolve_verified_managed_codex",
            lambda *_args, **_kwargs: managed,
        )
        verified = broker.VerifiedCodex(
            "/signed/codex", "0.149.0-alpha.4.3", ()
        )
        try:
            ready = broker.CodexAppServerDaemonController(codex_home).ensure_ready(
                verified
            )
        finally:
            endpoint.close()

        assert ready.socket_path == str(socket_path)
        assert [call[0][-2:] for call in calls] == [
            ["daemon", "start"],
            ["daemon", "version"],
        ]
        assert calls[0][1]["env"]["CODEX_HOME"] == str(codex_home)
        assert calls[0][1]["env"]["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"


def test_daemon_controller_accepts_allowlisted_signed_managed_peer_with_launcher_skew(
    monkeypatch,
) -> None:
    """The embedded CLI controls lifecycle; the managed binary owns the socket."""

    with tempfile.TemporaryDirectory(prefix="hcua-", dir="/tmp") as root:
        codex_home = Path(root) / "codex-home"
        control = codex_home / "app-server-control"
        control.mkdir(parents=True, mode=0o700)
        socket_path = control / broker.APP_SERVER_SOCKET_NAME
        endpoint = socket.socket(socket.AF_UNIX)
        endpoint.bind(str(socket_path))
        socket_path.chmod(0o600)
        launcher = broker.VerifiedCodex(
            "/Applications/ChatGPT.app/Contents/Resources/codex",
            "0.150.0-alpha.1",
            (),
        )
        managed = broker.VerifiedCodex(
            "/Users/victor/.codex/packages/standalone/releases/"
            "0.149.0-alpha.4.3-aarch64-apple-darwin/bin/codex",
            "0.149.0-alpha.4.3",
            (),
        )
        reported_path = codex_home / "packages/standalone/current/codex"
        resolved = []

        def fake_run(argv, **kwargs):
            if argv[-1] == "version":
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        '{"status":"running","backend":"pid",'
                        f'"managedCodexPath":"{reported_path}",'
                        '"managedCodexVersion":"0.149.0-alpha.4.3",'
                        f'"socketPath":"{socket_path}",'
                        '"cliVersion":"0.150.0-alpha.1",'
                        '"appServerVersion":"0.149.0-alpha.4.3"}'
                    ),
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")

        def fake_resolve(path, version, home, *, deadline=None):
            resolved.append((path, version, home, deadline))
            return managed

        monkeypatch.setattr(broker.subprocess, "run", fake_run)
        monkeypatch.setattr(
            broker, "_resolve_verified_managed_codex", fake_resolve, raising=False
        )
        try:
            ready = broker.CodexAppServerDaemonController(codex_home).ensure_ready(
                launcher
            )
        finally:
            endpoint.close()

        assert ready.version == "0.149.0-alpha.4.3"
        assert ready.verified == managed
        assert resolved == [
            (str(reported_path), "0.149.0-alpha.4.3", codex_home, None)
        ]


def test_managed_peer_resolution_rejects_signed_binary_version_mismatch(
    monkeypatch, tmp_path
) -> None:
    version = "0.149.0-alpha.4.3"
    codex_home = tmp_path / "codex-home"
    standalone = codex_home / "packages" / "standalone"
    release = standalone / "releases" / f"{version}-aarch64-apple-darwin"
    actual = release / "bin" / "codex"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"signed fixture")
    actual.chmod(0o755)
    (release / "codex").symlink_to("bin/codex")
    (standalone / "current").symlink_to(release, target_is_directory=True)
    alias = standalone / "current" / "codex"

    monkeypatch.setattr(broker.os, "uname", lambda: SimpleNamespace(machine="arm64"))
    monkeypatch.setattr(broker, "_trusted_path_snapshot", lambda _path: ())
    monkeypatch.setattr(
        broker,
        "_codesign_identity",
        lambda *_args, **_kwargs: (broker.CODEX_CLI_IDENTIFIER, broker.OPENAI_TEAM_ID),
    )
    monkeypatch.setattr(
        broker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="codex-cli 0.150.0", stderr=""
        ),
    )

    with pytest.raises(broker.CodexCUAVersionMismatch, match="binary version"):
        broker._resolve_verified_managed_codex(
            str(alias), version, codex_home
        )


def test_managed_peer_resolution_snapshots_official_alias_and_release(
    monkeypatch, tmp_path
) -> None:
    version = "0.149.0-alpha.4.3"
    codex_home = tmp_path / "codex-home"
    standalone = codex_home / "packages" / "standalone"
    release = standalone / "releases" / f"{version}-aarch64-apple-darwin"
    actual = release / "bin" / "codex"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(b"signed fixture")
    actual.chmod(0o755)
    package_alias = release / "codex"
    package_alias.symlink_to("bin/codex")
    current_alias = standalone / "current"
    current_alias.symlink_to(release, target_is_directory=True)

    monkeypatch.setattr(broker.os, "uname", lambda: SimpleNamespace(machine="arm64"))
    monkeypatch.setattr(broker, "_trusted_path_snapshot", lambda _path: ())
    monkeypatch.setattr(
        broker,
        "_codesign_identity",
        lambda *_args, **_kwargs: (broker.CODEX_CLI_IDENTIFIER, broker.OPENAI_TEAM_ID),
    )
    monkeypatch.setattr(
        broker.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=f"codex-cli {version}", stderr=""
        ),
    )

    verified = broker._resolve_verified_managed_codex(
        str(current_alias / "codex"), version, codex_home
    )

    assert verified.path == str(actual)
    assert verified.version == version
    current_alias.unlink()
    current_alias.symlink_to(release, target_is_directory=True)
    with pytest.raises(broker.CodexCUABinaryUntrusted, match="changed"):
        verified.recheck()


def test_daemon_controller_rejects_live_observed_0147_before_peer_resolution(
    monkeypatch, tmp_path
) -> None:
    codex_home = tmp_path / "codex-home"
    control = codex_home / "app-server-control"
    control.mkdir(parents=True, mode=0o700)
    socket_path = control / broker.APP_SERVER_SOCKET_NAME
    reported_path = codex_home / "packages/standalone/current/codex"
    resolved = []

    def fake_run(argv, **kwargs):
        if argv[-1] == "version":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    '{"status":"running","backend":"pid",'
                    f'"managedCodexPath":"{reported_path}",'
                    '"managedCodexVersion":"0.147.0",'
                    f'"socketPath":"{socket_path}",'
                    '"cliVersion":"0.149.0-alpha.4.3",'
                    '"appServerVersion":"0.147.0"}'
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(broker.subprocess, "run", fake_run)
    monkeypatch.setattr(
        broker,
        "_resolve_verified_managed_codex",
        lambda *_args, **_kwargs: resolved.append(True),
    )
    launcher = broker.VerifiedCodex(
        "/Applications/ChatGPT.app/Contents/Resources/codex",
        "0.149.0-alpha.4.3",
        (),
    )

    with pytest.raises(broker.CodexCUAVersionMismatch, match="unsupported"):
        broker.CodexAppServerDaemonController(codex_home).ensure_ready(launcher)
    assert resolved == []
