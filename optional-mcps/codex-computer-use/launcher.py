#!/usr/bin/env python3
"""Verify and launch the locally installed OpenAI Computer Use MCP.

This adapter contains no OpenAI code or assets. It resolves the signed client
that Codex installed under ``CODEX_HOME``, verifies both app bundles against
their exact Apple code-signing requirements, and only then replaces itself
with the MCP client process.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Mapping, NamedTuple


TEAM_ID = "2DC432GLL2"
SERVICE_BUNDLE_ID = "com.openai.sky.CUAService"
CLIENT_BUNDLE_ID = "com.openai.sky.CUAService.cli"
_CODESIGN = "/usr/bin/codesign"


class LauncherError(RuntimeError):
    """The installed client cannot be trusted or launched safely."""


class Installation(NamedTuple):
    codex_home: Path
    service_bundle: Path
    client_bundle: Path
    executable: Path


def resolve_installation(
    *, env: Mapping[str, str] | None = None, home: Path | None = None
) -> Installation:
    """Resolve stable OpenAI CUA paths without assuming a username."""

    environment = os.environ if env is None else env
    configured = str(environment.get("CODEX_HOME") or "").strip()
    codex_home = (
        Path(configured).expanduser()
        if configured
        else (home or Path.home()) / ".codex"
    )
    codex_home = codex_home.absolute()
    service = codex_home / "computer-use" / "Codex Computer Use.app"
    client = (
        service / "Contents" / "SharedSupport" / "SkyComputerUseClient.app"
    )
    executable = (
        client / "Contents" / "MacOS" / "SkyComputerUseClient"
    )
    return Installation(codex_home, service, client, executable)


def _designated_requirement(bundle_id: str) -> str:
    return (
        f'=identifier "{bundle_id}" and anchor apple generic and '
        "certificate 1[field.1.2.840.113635.100.6.2.6] exists and "
        "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and "
        f'certificate leaf[subject.OU] = "{TEAM_ID}"'
    )


def _verify_bundle(label: str, path: Path, bundle_id: str, *, run) -> None:
    if not path.is_dir():
        raise LauncherError(f"Expected {label} bundle is unavailable.")
    try:
        result = run(
            [
                _CODESIGN,
                "--verify",
                "--deep",
                "--strict",
                "--verbose=2",
                "-R",
                _designated_requirement(bundle_id),
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        raise LauncherError(f"{label.title()} signature verification failed.") from None
    if result.returncode != 0:
        raise LauncherError(f"{label.title()} signature verification failed.")


def verify_installation(
    installation: Installation,
    *,
    run=subprocess.run,
    platform: str | None = None,
) -> None:
    """Fail closed unless the expected signed service and client are intact."""

    host = sys.platform if platform is None else platform
    if host != "darwin":
        raise LauncherError("Codex Computer Use requires macOS.")
    if not installation.executable.is_file() or not os.access(
        installation.executable, os.X_OK
    ):
        raise LauncherError("Expected client executable is unavailable.")
    snapshot_installation(installation)
    _verify_bundle(
        "service",
        installation.service_bundle,
        SERVICE_BUNDLE_ID,
        run=run,
    )
    _verify_bundle(
        "client",
        installation.client_bundle,
        CLIENT_BUNDLE_ID,
        run=run,
    )


def _installation_components(installation: Installation) -> tuple[Path, ...]:
    relative = (
        Path(),
        Path("computer-use"),
        Path("computer-use/Codex Computer Use.app"),
        Path("computer-use/Codex Computer Use.app/Contents"),
        Path("computer-use/Codex Computer Use.app/Contents/SharedSupport"),
        Path(
            "computer-use/Codex Computer Use.app/Contents/SharedSupport/"
            "SkyComputerUseClient.app"
        ),
        Path(
            "computer-use/Codex Computer Use.app/Contents/SharedSupport/"
            "SkyComputerUseClient.app/Contents"
        ),
        Path(
            "computer-use/Codex Computer Use.app/Contents/SharedSupport/"
            "SkyComputerUseClient.app/Contents/MacOS"
        ),
        Path(
            "computer-use/Codex Computer Use.app/Contents/SharedSupport/"
            "SkyComputerUseClient.app/Contents/MacOS/SkyComputerUseClient"
        ),
    )
    return tuple(installation.codex_home / part for part in relative)


def _identity(path: Path) -> tuple[int, ...]:
    try:
        info = path.lstat()
    except OSError:
        raise LauncherError("Expected installation component is unavailable.") from None
    if stat.S_ISLNK(info.st_mode):
        raise LauncherError("Expected installation path contains a symbolic link.")
    current_uid = os.getuid()  # windows-footgun: ok -- macOS-only launcher
    if info.st_uid != current_uid or info.st_mode & 0o022:
        raise LauncherError("Expected installation ownership or mode is unsafe.")
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def snapshot_installation(installation: Installation) -> dict[str, tuple[int, ...]]:
    """Capture every canonical path component before signature verification."""

    return {str(path): _identity(path) for path in _installation_components(installation)}


def recheck_installation(
    installation: Installation, snapshot: dict[str, tuple[int, ...]]
) -> None:
    """Detect path substitution or mutation immediately before exec."""

    current = snapshot_installation(installation)
    if current != snapshot:
        raise LauncherError("Installed client changed after verification.")


def build_exec_argv(installation: Installation) -> tuple[str, str]:
    return str(installation.executable), "mcp"


def main(*, execve=None) -> int:
    try:
        installation = resolve_installation()
    except (OSError, RuntimeError):
        print(
            "Codex Computer Use MCP unavailable: installation could not be resolved.",
            file=sys.stderr,
        )
        return 78

    try:
        snapshot = snapshot_installation(installation)
        verify_installation(installation)
        recheck_installation(installation, snapshot)
    except LauncherError as exc:
        print(f"Codex Computer Use MCP unavailable: {exc}", file=sys.stderr)
        return 78

    executable, mode = build_exec_argv(installation)
    child_env = {
        key: os.environ[key]
        for key in (
            "HOME",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
        )
        if key in os.environ
    }
    child_env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    child_env["CODEX_HOME"] = str(installation.codex_home)
    replace_process = os.execve if execve is None else execve
    try:
        replace_process(executable, [executable, mode], child_env)
    except OSError:
        print(
            "Codex Computer Use MCP unavailable: verified client could not be started.",
            file=sys.stderr,
        )
        return 78
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
