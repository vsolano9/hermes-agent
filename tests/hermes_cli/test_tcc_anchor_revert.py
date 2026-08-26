"""Tests for the TCC-anchor revert heal (#95425 / #95541).

The interpreter anchor replaced venv/bin/python with a real-file copy that
could not load libpython on real Macs, bricking the CLI. The anchor is
reverted; doctor's check_macos_tcc_anchor_removed() restores anchored venvs
to symlinks using the marker the anchor left behind.
"""

import contextlib
import io
import os
from pathlib import Path

import hermes_cli.doctor as doctor_mod


def _capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def _build_anchored_checkout(tmp_path):
    """A checkout whose venv the anchor converted: real-file python + marker."""
    root = tmp_path / "checkout"
    store_bin = (
        tmp_path
        / "uv"
        / "python"
        / "cpython-3.12.1-macos-aarch64-none"
        / "bin"
    )
    store_bin.mkdir(parents=True)
    source = store_bin / "python3.12"
    source.write_bytes(b"#!store interpreter")
    source.chmod(0o755)
    venv_bin = root / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_py = venv_bin / "python"
    venv_py.write_bytes(b"#!anchored copy (broken on real macs)")
    venv_py.chmod(0o755)
    (root / "venv" / "pyvenv.cfg").write_text(
        f"home = {store_bin}\n"
        "implementation = CPython\n"
        "uv = 0.12.5\n"
        "version_info = 3.12.1\n",
        encoding="utf-8",
    )
    (venv_bin / ".tcc-anchor-source").write_text(str(source), encoding="utf-8")
    os.symlink(venv_py, venv_bin / "python3")
    os.symlink(venv_py, venv_bin / "python3.12")
    return root, source, venv_py


def test_silent_on_non_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "linux")
    assert _capture(doctor_mod.check_macos_tcc_anchor_removed) == ""


def test_silent_when_never_anchored(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root = tmp_path / "checkout"
    (root / "venv" / "bin").mkdir(parents=True)
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert out == ""


def test_heals_anchored_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)

    # Point the check's root resolution at the fixture checkout.
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "TCC anchor removed" in out
    assert venv_py.is_symlink()
    assert Path(os.readlink(venv_py)) == source
    assert not (venv_py.parent / ".tcc-anchor-source").exists()
    # Aliases restored to point at bin/python.
    alias = venv_py.parent / "python3"
    assert alias.is_symlink()
    assert os.readlink(alias) == "python"


def test_stale_marker_fails_closed_and_preserves_recovery_state(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)
    marker = venv_py.parent / ".tcc-anchor-source"
    source.unlink()
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "cleanup failed" in out
    assert "TCC anchor removed" not in out
    assert marker.is_file()
    assert not venv_py.is_symlink()


def test_partial_retry_completes_owned_aliases_before_removing_marker(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)
    marker = venv_py.parent / ".tcc-anchor-source"
    venv_py.unlink()
    os.symlink(source, venv_py)
    # Simulate an interrupted first attempt: python is healed, aliases are not.
    assert os.readlink(venv_py.parent / "python3") != "python"
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "TCC anchor removed" in out
    assert not marker.exists()
    assert os.readlink(venv_py) == str(source)
    assert os.readlink(venv_py.parent / "python3") == "python"
    assert os.readlink(venv_py.parent / "python3.12") == "python"
    # A second run is a silent no-op after the completed transaction.
    assert _capture(doctor_mod.check_macos_tcc_anchor_removed) == ""


def test_interrupted_alias_restore_keeps_marker_and_retry_completes(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)
    marker = venv_py.parent / ".tcc-anchor-source"
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )
    real_replace = os.replace

    def fail_python3_once(source_path, destination):
        if Path(destination).name == "python3":
            raise OSError("simulated interruption")
        return real_replace(source_path, destination)

    monkeypatch.setattr(os, "replace", fail_python3_once)
    first = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "cleanup failed" in first
    assert marker.is_file()
    assert venv_py.is_symlink()
    assert os.readlink(venv_py) == str(source)

    monkeypatch.setattr(os, "replace", real_replace)
    second = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "TCC anchor removed" in second
    assert not marker.exists()
    assert os.readlink(venv_py.parent / "python3") == "python"
    assert os.readlink(venv_py.parent / "python3.12") == "python"


def test_untrusted_marker_target_is_rejected_without_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, _source, venv_py = _build_anchored_checkout(tmp_path)
    marker = venv_py.parent / ".tcc-anchor-source"
    attacker = tmp_path / "attacker-python"
    attacker.write_bytes(b"#!not a uv interpreter")
    attacker.chmod(0o755)
    marker.write_text(str(attacker), encoding="utf-8")
    original = venv_py.read_bytes()
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "cleanup failed" in out
    assert "TCC anchor removed" not in out
    assert marker.read_text(encoding="utf-8") == str(attacker)
    assert not venv_py.is_symlink()
    assert venv_py.read_bytes() == original


def test_only_owned_interpreter_aliases_are_replaced(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, _source, venv_py = _build_anchored_checkout(tmp_path)
    unrelated_config = venv_py.parent / "python3-config"
    unrelated_config.write_text("do not replace", encoding="utf-8")
    unrelated_alias = venv_py.parent / "python3.99"
    os.symlink("some-other-python", unrelated_alias)
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "TCC anchor removed" in out
    assert unrelated_config.read_text(encoding="utf-8") == "do not replace"
    assert not unrelated_config.is_symlink()
    assert os.readlink(unrelated_alias) == "some-other-python"


def test_missing_marker_on_uv_anchor_warns_without_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, _source, venv_py = _build_anchored_checkout(tmp_path)
    marker = venv_py.parent / ".tcc-anchor-source"
    marker.unlink()
    original = venv_py.read_bytes()
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "recovery marker is missing" in out
    assert "TCC anchor removed" not in out
    assert not venv_py.is_symlink()
    assert venv_py.read_bytes() == original


def test_symlinked_uv_source_is_rejected_without_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)
    marker = venv_py.parent / ".tcc-anchor-source"
    real_source = source.with_name("real-python3.12")
    source.replace(real_source)
    os.symlink(real_source, source)
    original = venv_py.read_bytes()
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "cleanup failed" in out
    assert marker.is_file()
    assert not venv_py.is_symlink()
    assert venv_py.read_bytes() == original


def test_insecure_uv_source_mode_is_rejected_without_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)
    marker = venv_py.parent / ".tcc-anchor-source"
    source.chmod(0o775)
    original = venv_py.read_bytes()
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "cleanup failed" in out
    assert marker.is_file()
    assert not venv_py.is_symlink()
    assert venv_py.read_bytes() == original


def test_accepts_exact_pyvenv_home_alias_spelling(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)
    install = source.parent.parent
    install_alias = install.with_name("cpython-3.12-macos-aarch64-none")
    os.symlink(install, install_alias)
    alias_source = install_alias / "bin" / source.name
    (root / "venv" / "pyvenv.cfg").write_text(
        f"home = {install_alias / 'bin'}\n"
        "implementation = CPython\n"
        "uv = 0.12.5\n"
        "version_info = 3.12.1\n",
        encoding="utf-8",
    )
    (venv_py.parent / ".tcc-anchor-source").write_text(
        str(alias_source), encoding="utf-8"
    )
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "TCC anchor removed" in out
    assert os.readlink(venv_py) == str(source)


def test_accepts_resolved_source_for_symlinked_pyvenv_home(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)
    install = source.parent.parent
    install_alias = install.with_name("cpython-3.12-macos-aarch64-none")
    os.symlink(install, install_alias)
    (root / "venv" / "pyvenv.cfg").write_text(
        f"home = {install_alias / 'bin'}\n"
        "implementation = CPython\n"
        "uv = 0.12.5\n"
        "version_info = 3.12.1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "TCC anchor removed" in out
    assert os.readlink(venv_py) == str(source)


def test_rejects_noncanonical_alias_spelling_even_when_target_is_trusted(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(doctor_mod.sys, "platform", "darwin")
    root, source, venv_py = _build_anchored_checkout(tmp_path)
    arbitrary_root = tmp_path / "arbitrary-uv-alias"
    os.symlink(tmp_path / "uv", arbitrary_root)
    arbitrary_source = arbitrary_root / source.relative_to(tmp_path / "uv")
    marker = venv_py.parent / ".tcc-anchor-source"
    marker.write_text(str(arbitrary_source), encoding="utf-8")
    original = venv_py.read_bytes()
    monkeypatch.setattr(
        doctor_mod, "__file__", str(root / "hermes_cli" / "doctor.py")
    )

    out = _capture(doctor_mod.check_macos_tcc_anchor_removed)

    assert "cleanup failed" in out
    assert marker.is_file()
    assert not venv_py.is_symlink()
    assert venv_py.read_bytes() == original
