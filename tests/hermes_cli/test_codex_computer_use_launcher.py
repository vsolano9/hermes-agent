"""Behavior tests for the local Codex Computer Use MCP launcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


LAUNCHER_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-mcps"
    / "codex-computer-use"
    / "launcher.py"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "codex_computer_use_launcher", LAUNCHER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_installation(tmp_path: Path):
    codex_home = tmp_path / "portable-codex-home"
    service = codex_home / "computer-use" / "Codex Computer Use.app"
    client = service / "Contents" / "SharedSupport" / "SkyComputerUseClient.app"
    executable = client / "Contents" / "MacOS" / "SkyComputerUseClient"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fake executable")
    executable.chmod(0o755)
    return codex_home, service, client, executable


def test_resolve_installation_prefers_configured_codex_home(tmp_path):
    launcher = _load_launcher()
    codex_home, service, client, executable = _make_installation(tmp_path)

    resolved = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path / "ignored-home"
    )

    assert resolved.codex_home == codex_home.resolve()
    assert resolved.service_bundle == service.resolve()
    assert resolved.client_bundle == client.resolve()
    assert resolved.executable == executable.resolve()


def test_resolve_installation_defaults_to_user_codex_home(tmp_path):
    launcher = _load_launcher()
    canonical_home = tmp_path / ".codex"
    canonical_executable = (
        canonical_home
        / "computer-use"
        / "Codex Computer Use.app"
        / "Contents"
        / "SharedSupport"
        / "SkyComputerUseClient.app"
        / "Contents"
        / "MacOS"
        / "SkyComputerUseClient"
    )
    canonical_executable.parent.mkdir(parents=True, exist_ok=True)
    canonical_executable.write_bytes(b"fake executable")
    canonical_executable.chmod(0o755)

    resolved = launcher.resolve_installation(env={}, home=tmp_path)

    assert resolved.codex_home == canonical_home.resolve()
    assert resolved.executable == canonical_executable.resolve()


def test_verify_installation_checks_both_signed_bundle_identities(tmp_path):
    launcher = _load_launcher()
    codex_home, service, client, _executable = _make_installation(tmp_path)
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path
    )
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    launcher.verify_installation(installation, run=fake_run, platform="darwin")

    assert len(calls) == 2
    assert calls[0][0][-1] == str(service.resolve())
    assert calls[1][0][-1] == str(client.resolve())
    assert "com.openai.sky.CUAService" in calls[0][0][-2]
    assert "com.openai.sky.CUAService.cli" in calls[1][0][-2]
    for argv, kwargs in calls:
        requirement = argv[-2]
        assert argv[:5] == [
            "/usr/bin/codesign", "--verify", "--deep", "--strict", "--verbose=2"
        ]
        assert argv[5] == "-R"
        assert "=identifier" in requirement
        assert "anchor apple generic" in requirement
        assert "certificate 1[field.1.2.840.113635.100.6.2.6] exists" in requirement
        assert "certificate leaf[field.1.2.840.113635.100.6.1.13] exists" in requirement
        assert "2DC432GLL2" in requirement
        assert kwargs["shell"] is False


def test_verify_installation_fails_closed_on_signature_mismatch(tmp_path):
    launcher = _load_launcher()
    codex_home, _service, _client, _executable = _make_installation(tmp_path)
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path
    )
    attempt = 0

    def fake_run(_argv, **_kwargs):
        nonlocal attempt
        attempt += 1
        return SimpleNamespace(
            returncode=0 if attempt == 1 else 1,
            stdout="",
            stderr="sensitive codesign diagnostic that must not escape",
        )

    with pytest.raises(launcher.LauncherError) as exc_info:
        launcher.verify_installation(installation, run=fake_run, platform="darwin")

    message = str(exc_info.value)
    assert "client signature verification failed" in message.lower()
    assert "sensitive codesign diagnostic" not in message
    assert str(codex_home) not in message


def test_verify_installation_fails_closed_when_expected_executable_disappears(
    tmp_path,
):
    launcher = _load_launcher()
    codex_home, _service, _client, executable = _make_installation(tmp_path)
    executable.unlink()
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path
    )

    with pytest.raises(launcher.LauncherError) as exc_info:
        launcher.verify_installation(
            installation,
            run=lambda *_a, **_kw: pytest.fail("codesign must not run"),
            platform="darwin",
        )

    assert "expected client executable is unavailable" in str(exc_info.value).lower()
    assert str(codex_home) not in str(exc_info.value)


def test_verify_installation_rejects_non_macos_hosts(tmp_path):
    launcher = _load_launcher()
    codex_home, _service, _client, _executable = _make_installation(tmp_path)
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path
    )

    with pytest.raises(launcher.LauncherError, match="requires macOS"):
        launcher.verify_installation(
            installation,
            run=lambda *_a, **_kw: pytest.fail("codesign must not run"),
            platform="linux",
        )


def test_build_exec_argv_uses_verified_client_and_mcp_mode(tmp_path):
    launcher = _load_launcher()
    codex_home, _service, _client, executable = _make_installation(tmp_path)
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path
    )

    assert launcher.build_exec_argv(installation) == (
        str(executable.resolve()),
        "mcp",
    )


def test_main_sanitizes_path_resolution_errors(monkeypatch, capsys):
    launcher = _load_launcher()
    secret_path = "/private/sensitive/codex-home"
    monkeypatch.setattr(
        launcher,
        "resolve_installation",
        lambda: (_ for _ in ()).throw(OSError(secret_path)),
    )

    result = launcher.main()

    captured = capsys.readouterr()
    assert result == 78
    assert "installation could not be resolved" in captured.err.lower()
    assert secret_path not in captured.err


def test_main_sanitizes_exec_failure_after_verification(monkeypatch, capsys, tmp_path):
    launcher = _load_launcher()
    codex_home, service, client, executable = _make_installation(tmp_path)
    installation = launcher.Installation(codex_home, service, client, executable)
    monkeypatch.setattr(launcher, "resolve_installation", lambda: installation)
    monkeypatch.setattr(launcher, "verify_installation", lambda _installation: None)
    secret_error = f"failed to execute {executable}"

    result = launcher.main(
        execve=lambda *_args: (_ for _ in ()).throw(OSError(secret_error))
    )

    captured = capsys.readouterr()
    assert result == 78
    assert "verified client could not be started" in captured.err.lower()
    assert secret_error not in captured.err


def test_symlinked_installation_is_rejected_before_codesign(tmp_path):
    launcher = _load_launcher()
    real_home, _service, _client, _executable = _make_installation(tmp_path / "real")
    linked_home = tmp_path / "linked-codex-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(linked_home)}, home=tmp_path
    )

    with pytest.raises(launcher.LauncherError, match="symbolic link"):
        launcher.verify_installation(
            installation,
            run=lambda *_a, **_kw: pytest.fail("codesign must not run"),
            platform="darwin",
        )


@pytest.mark.parametrize(
    "relative_component",
    [
        "computer-use",
        "computer-use/Codex Computer Use.app/Contents",
        "computer-use/Codex Computer Use.app/Contents/SharedSupport",
        (
            "computer-use/Codex Computer Use.app/Contents/SharedSupport/"
            "SkyComputerUseClient.app/Contents"
        ),
        (
            "computer-use/Codex Computer Use.app/Contents/SharedSupport/"
            "SkyComputerUseClient.app/Contents/MacOS"
        ),
    ],
)
def test_nested_symlink_component_is_rejected_before_codesign(
    tmp_path, relative_component
):
    launcher = _load_launcher()
    codex_home, _service, _client, _executable = _make_installation(tmp_path)
    component = codex_home / relative_component
    target = component.with_name(component.name + "-real")
    component.rename(target)
    component.symlink_to(target, target_is_directory=True)
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path
    )

    with pytest.raises(launcher.LauncherError, match="symbolic link"):
        launcher.verify_installation(
            installation,
            run=lambda *_a, **_kw: pytest.fail("codesign must not run"),
            platform="darwin",
        )


def test_identity_snapshot_detects_executable_replacement(tmp_path):
    launcher = _load_launcher()
    codex_home, _service, _client, executable = _make_installation(tmp_path)
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path
    )
    snapshot = launcher.snapshot_installation(installation)
    executable.unlink()
    executable.write_bytes(b"replacement")
    executable.chmod(0o755)

    with pytest.raises(launcher.LauncherError, match="changed after verification"):
        launcher.recheck_installation(installation, snapshot)


def test_identity_snapshot_detects_nested_bundle_tamper(tmp_path):
    launcher = _load_launcher()
    codex_home, _service, client, _executable = _make_installation(tmp_path)
    installation = launcher.resolve_installation(
        env={"CODEX_HOME": str(codex_home)}, home=tmp_path
    )
    snapshot = launcher.snapshot_installation(installation)
    (client / "tamper").write_text("x", encoding="utf-8")

    with pytest.raises(launcher.LauncherError, match="changed after verification"):
        launcher.recheck_installation(installation, snapshot)


def test_main_passes_only_minimal_environment_to_verified_child(
    monkeypatch, tmp_path
):
    launcher = _load_launcher()
    codex_home, service, client, executable = _make_installation(tmp_path)
    installation = launcher.Installation(codex_home, service, client, executable)
    captured = {}
    monkeypatch.setenv("SECRET_CANARY", "must-not-propagate")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(launcher, "resolve_installation", lambda: installation)
    monkeypatch.setattr(launcher, "verify_installation", lambda _installation: None)

    def fake_execve(_executable, _argv, env):
        captured.update(env)
        raise OSError("stop")

    launcher.main(execve=fake_execve)

    assert captured["CODEX_HOME"] == str(codex_home)
    assert captured["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert "SECRET_CANARY" not in captured


def test_main_final_recheck_blocks_mutation_between_codesign_and_exec(
    monkeypatch, tmp_path
):
    launcher = _load_launcher()
    codex_home, service, client, executable = _make_installation(tmp_path)
    installation = launcher.Installation(codex_home, service, client, executable)
    monkeypatch.setattr(launcher, "resolve_installation", lambda: installation)

    def mutate_after_verification(_installation):
        executable.unlink()
        executable.write_bytes(b"post-codesign replacement")
        executable.chmod(0o755)

    monkeypatch.setattr(launcher, "verify_installation", mutate_after_verification)
    called = []

    result = launcher.main(execve=lambda *_args: called.append(True))

    assert result == 78
    assert called == []
