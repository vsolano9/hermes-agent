"""Cross-process plugin registration must be serialized per profile/plugin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import (
    PluginManager,
    PluginState,
    _acquire_plugin_registration_fd,
    _locked_plugin_registration,
    _plugin_registration_lock_directory,
)


_WORKER = r"""
import json
import sys
import time
from pathlib import Path

from hermes_cli.plugins import PluginManager

barrier = Path(sys.argv[1])
worker_id = sys.argv[2]
(barrier / f"ready-{worker_id}").write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 10
while not (barrier / "go").exists():
    if time.monotonic() >= deadline:
        raise SystemExit("barrier timeout")
    time.sleep(0.005)

manager = PluginManager()
manager.discover_and_load()
loaded = manager._plugins.get("registration-race")
(barrier / f"result-{worker_id}.json").write_text(
    json.dumps(
        {
            "enabled": bool(loaded and loaded.enabled),
            "error": loaded.error if loaded else "plugin missing",
            "data_dir": str(manager._plugins["registration-race"].module.DATA_DIR)
            if loaded and loaded.enabled
            else None,
        }
    ),
    encoding="utf-8",
)
"""


_TIMEOUT_WORKER = r"""
import json

import hermes_cli.plugins as plugins

plugins._PLUGIN_REGISTRATION_LOCK_TIMEOUT_SECONDS = 0.2
manager = plugins.PluginManager()
manager.discover_and_load()
loaded = manager._plugins.get("registration-race")
print(json.dumps({"enabled": loaded.enabled, "error": loaded.error}))
"""


def _write_overlap_detecting_plugin(home: Path) -> None:
    plugin = home / "plugins" / "registration-race"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "registration-race",
                "version": "1.0.0",
                "description": "Detect overlapping process registration",
            }
        ),
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        """from __future__ import annotations

import os
import time

DATA_DIR = None


def register(ctx):
    global DATA_DIR
    DATA_DIR = ctx.state.data_dir
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    active = DATA_DIR / "active-registration"
    try:
        fd = os.open(active, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("overlapping cross-process plugin registration") from exc
    try:
        os.close(fd)
        time.sleep(0.4)
        ctx.state.set(f"registered_{os.getpid()}", True)
    finally:
        active.unlink(missing_ok=True)
""",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["registration-race"]}}),
        encoding="utf-8",
    )


def test_four_panes_and_gateway_serialize_plugin_registration(tmp_path: Path) -> None:
    """Four panes plus the gateway share one stable plugin namespace.

    The start barrier makes all five processes enter discovery together.  The
    fixture plugin fails closed if two ``register`` calls overlap, matching the
    storage invariant enforced by multi-model-delegation.
    """
    home = tmp_path / "home"
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    _write_overlap_detecting_plugin(home)
    env = dict(os.environ, HERMES_HOME=str(home))
    repo_root = Path(__file__).resolve().parents[2]
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(barrier), str(index)],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(5)
    ]
    try:
        deadline = time.monotonic() + 10
        while len(list(barrier.glob("ready-*"))) != 5:
            assert time.monotonic() < deadline, "workers did not reach start barrier"
            time.sleep(0.005)
        (barrier / "go").write_text("go", encoding="utf-8")

        diagnostics = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=12)
            diagnostics.append((process.returncode, stdout, stderr))
        assert [item[0] for item in diagnostics] == [0] * 5, diagnostics
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)

    results = [
        json.loads((barrier / f"result-{index}.json").read_text(encoding="utf-8"))
        for index in range(5)
    ]
    assert all(result["enabled"] for result in results), results
    assert {result["error"] for result in results} == {None}
    assert len({result["data_dir"] for result in results}) == 1
    state = json.loads(
        next((home / "plugin-data").glob("*/state.json")).read_text(encoding="utf-8")
    )
    assert len([key for key in state if key.startswith("registered_")]) == 5
    assert not list((home / "plugin-data").glob("*/active-registration"))


def test_registration_lock_timeout_fails_plugin_closed_without_deadlock(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    _write_overlap_detecting_plugin(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = PluginState("registration-race")
    env = dict(os.environ, HERMES_HOME=str(home))

    with _locked_plugin_registration(state):
        completed = subprocess.run(
            [sys.executable, "-c", _TIMEOUT_WORKER],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            capture_output=True,
            text=True,
            timeout=4,
            check=True,
        )

    result = json.loads(completed.stdout)
    assert result["enabled"] is False
    assert result["error"] == "Timed out acquiring plugin registration lock."


def _registration_lock_path(state: PluginState) -> Path:
    return _plugin_registration_lock_directory() / f"{state._data_namespace}.lock"


def test_registration_lock_ignores_umask_and_uses_private_modes(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = PluginState("registration-race")
    previous_umask = os.umask(0)
    try:
        with _locked_plugin_registration(state):
            lock_path = _registration_lock_path(state)
    finally:
        os.umask(previous_umask)

    assert lock_path.stat().st_mode & 0o777 == 0o600
    assert lock_path.parent.stat().st_mode & 0o777 == 0o700
    assert lock_path.parent.parent.stat().st_mode & 0o777 == 0o700


def test_registration_lock_rejects_preplanted_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = PluginState("registration-race")
    lock_path = _registration_lock_path(state)
    target = tmp_path / "attacker-lock"
    target.write_text("attacker", encoding="utf-8")
    lock_path.symlink_to(target)

    with pytest.raises(RuntimeError, match="lock file is unavailable"):
        with _locked_plugin_registration(state):
            raise AssertionError("symlink lock must not be acquired")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_registration_lock_rejects_nonregular_fifo(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = PluginState("registration-race")
    lock_path = _registration_lock_path(state)
    os.mkfifo(lock_path, 0o600)

    with pytest.raises(RuntimeError, match="lock file is invalid"):
        with _locked_plugin_registration(state):
            raise AssertionError("FIFO lock must not be acquired")


def test_registration_lock_rejects_hardlinked_inode(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = PluginState("registration-race")
    lock_path = _registration_lock_path(state)
    source = tmp_path / "shared-inode"
    source.write_text("shared", encoding="utf-8")
    os.link(source, lock_path)

    with pytest.raises(RuntimeError, match="lock file is invalid"):
        with _locked_plugin_registration(state):
            raise AssertionError("hardlinked lock must not be acquired")


def test_registration_lock_rejects_symlinked_host_directory(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (home / ".locks").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    with pytest.raises(RuntimeError, match="lock directory is invalid"):
        with _locked_plugin_registration(PluginState("registration-race")):
            raise AssertionError("symlinked host directory must not be used")


def test_registration_lock_rejects_permissive_host_directory(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    lock_root = home / ".locks"
    lock_root.mkdir(mode=0o755)
    monkeypatch.setenv("HERMES_HOME", str(home))

    with pytest.raises(RuntimeError, match="lock directory mode is invalid"):
        with _locked_plugin_registration(PluginState("registration-race")):
            raise AssertionError("permissive host directory must not be used")


def test_registration_lock_rejects_foreign_file_owner_injected(
    tmp_path: Path, monkeypatch
) -> None:
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX ownership unavailable")
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    real_fstat = os.fstat

    def foreign_fstat(fd: int):
        metadata = list(real_fstat(fd))
        metadata[4] = os.getuid() + 1
        return os.stat_result(metadata)

    monkeypatch.setattr(plugins_mod.os, "fstat", foreign_fstat)
    with pytest.raises(RuntimeError, match="lock file owner is invalid"):
        with _locked_plugin_registration(PluginState("registration-race")):
            raise AssertionError("foreign-owned lock must not be acquired")


def test_registration_lock_rejects_mode_when_fchmod_cannot_correct_it(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    state = PluginState("registration-race")
    lock_path = _registration_lock_path(state)
    lock_path.write_text("lock", encoding="utf-8")
    lock_path.chmod(0o644)
    monkeypatch.setattr(plugins_mod.os, "fchmod", lambda _fd, _mode: None)

    with pytest.raises(RuntimeError, match="lock file mode is invalid"):
        with _locked_plugin_registration(state):
            raise AssertionError("permissive lock must not be acquired")


def test_nested_manager_registration_reenters_without_second_os_lock(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    plugin = home / "plugins" / "nested-registration"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "nested-registration", "version": "1.0.0"}),
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        "    from hermes_cli.plugins import _locked_plugin_registration\n"
        "    with _locked_plugin_registration(ctx.state):\n"
        "        ctx.state.set('nested', True)\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["nested-registration"]}}),
        encoding="utf-8",
    )
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    acquired: list[int] = []
    released: list[int] = []
    real_acquire = plugins_mod._acquire_plugin_registration_fd
    real_release = plugins_mod._release_plugin_registration_fd

    def counted_acquire(fd: int, timeout_seconds: float, **kwargs) -> None:
        acquired.append(fd)
        real_acquire(fd, timeout_seconds, **kwargs)

    def counted_release(fd: int, **kwargs) -> None:
        released.append(fd)
        real_release(fd, **kwargs)

    monkeypatch.setattr(plugins_mod, "_acquire_plugin_registration_fd", counted_acquire)
    monkeypatch.setattr(plugins_mod, "_release_plugin_registration_fd", counted_release)
    manager = PluginManager()
    manager.discover_and_load()

    assert manager._plugins["nested-registration"].enabled is True
    assert len(acquired) == 1
    assert len(released) == 1


def test_reentrant_lock_releases_only_when_depth_reaches_zero(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    released: list[int] = []
    real_release = plugins_mod._release_plugin_registration_fd

    def counted_release(fd: int, **kwargs) -> None:
        released.append(fd)
        real_release(fd, **kwargs)

    monkeypatch.setattr(plugins_mod, "_release_plugin_registration_fd", counted_release)
    state = PluginState("registration-race")
    outer = _locked_plugin_registration(state)
    inner = _locked_plugin_registration(state)
    outer.__enter__()
    inner.__enter__()

    outer.__exit__(None, None, None)
    assert released == []
    inner.__exit__(None, None, None)
    assert len(released) == 1


def test_registration_lock_recovers_after_holder_process_crash(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    marker = tmp_path / "held"
    crash_worker = r"""
import os
from pathlib import Path
from hermes_cli.plugins import PluginState, _locked_plugin_registration
with _locked_plugin_registration(PluginState("registration-race")):
    Path(os.environ["LOCK_MARKER"]).write_text("held", encoding="utf-8")
    os._exit(23)
"""
    env = dict(os.environ, HERMES_HOME=str(home), LOCK_MARKER=str(marker))
    crashed = subprocess.run(
        [sys.executable, "-c", crash_worker],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        timeout=4,
        check=False,
    )
    assert crashed.returncode == 23
    assert marker.read_text(encoding="utf-8") == "held"

    with _locked_plugin_registration(PluginState("registration-race")):
        assert True


def test_windows_registration_lock_timeout_branch_is_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    lock_file = tmp_path / "windows.lock"
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o600)
    fake_msvcrt = types.SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=lambda _fd, _mode, _size: (_ for _ in ()).throw(OSError("busy")),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    try:
        with pytest.raises(TimeoutError, match="Timed out acquiring"):
            _acquire_plugin_registration_fd(fd, 0, platform_name="nt")
    finally:
        os.close(fd)
