"""Trust-boundary tests for the replacement Codex App Server broker."""

from __future__ import annotations

from pathlib import Path
import socket
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
    (codex_home / "app-server-control").mkdir(parents=True, mode=0o700)

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
                    stdout='{"managedCodexVersion":"0.149.0-alpha.4.3",'
                    '"appServerVersion":"0.149.0-alpha.4.3"}',
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="{}", stderr="")

        monkeypatch.setattr(broker.subprocess, "run", fake_run)
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
