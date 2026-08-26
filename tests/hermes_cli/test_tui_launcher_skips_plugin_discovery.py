
"""Regression test: the TUI launcher must not spend time on plugin discovery.

`hermes --tui` just spawns a Node process; the spawned tui_gateway backend
performs its own plugin discovery. Running discover_plugins() in the
launcher added ~0.5s to every `hermes --tui` startup for work the backend
then redoes. Plain chat must still discover plugins.
"""

from __future__ import annotations

from argparse import Namespace
import sys
import types

from hermes_cli import main as main_mod
from hermes_cli import mcp_startup


def _install_discover_spy(monkeypatch):
    calls = []

    def _discover():
        calls.append("discover")

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(
            discover_plugins=_discover,
            # main.py now kicks discovery off in a background thread; both
            # entry points count as "discovery work happened in the launcher".
            start_background_plugin_discovery=_discover,
        ),
    )
    return calls


def _install_mcp_and_hook_spies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mcp_startup,
        "start_background_mcp_discovery",
        lambda **_kwargs: calls.append("discover"),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.outbound_webhooks",
        types.SimpleNamespace(register_from_config=lambda *_args, **_kwargs: None),
    )
    return calls


def _args(**overrides):
    base = {
        "accept_hooks": False,
        "yolo": False,
        "safe_mode": False,
        "command": None,
        "query": None,
        "image": None,
        "cli": False,
        "tui": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_plugin_discovery_skipped_for_tui_launch(monkeypatch):
    calls = _install_discover_spy(monkeypatch)
    mcp_calls = _install_mcp_and_hook_spies(monkeypatch)
    main_mod._prepare_agent_startup(_args(tui=True))
    assert calls == [], (
        "Plugin discovery must not run in the TUI launcher: the spawned "
        "tui_gateway backend discovers plugins itself."
    )
    assert mcp_calls == []


def test_ambient_tui_launcher_skips_plugin_and_mcp_discovery(monkeypatch):
    """A configured TUI default must be classified before agent startup."""
    plugin_calls = _install_discover_spy(monkeypatch)
    mcp_calls = _install_mcp_and_hook_spies(monkeypatch)

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.delenv("HERMES_TUI", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"display": {"interface": "tui"}},
    )
    main_mod._prepare_agent_startup(_args(command="chat"))

    assert plugin_calls == []
    assert mcp_calls == []


def test_cli_flag_beats_ambient_tui_and_keeps_cli_discovery(monkeypatch):
    plugin_calls = _install_discover_spy(monkeypatch)
    mcp_calls = _install_mcp_and_hook_spies(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("HERMES_TUI", "1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"display": {"interface": "tui"}},
    )

    main_mod._prepare_agent_startup(_args(command="chat", cli=True))

    assert plugin_calls == ["discover"]
    assert mcp_calls == ["discover"]


def test_non_tty_ambient_tui_keeps_cli_discovery(monkeypatch):
    plugin_calls = _install_discover_spy(monkeypatch)
    mcp_calls = _install_mcp_and_hook_spies(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.delenv("HERMES_TUI", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"display": {"interface": "tui"}},
    )

    main_mod._prepare_agent_startup(_args(command="chat"))

    assert plugin_calls == ["discover"]
    assert mcp_calls == ["discover"]


def test_hermes_tui_env_skips_launcher_discovery(monkeypatch):
    plugin_calls = _install_discover_spy(monkeypatch)
    mcp_calls = _install_mcp_and_hook_spies(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("HERMES_TUI", "1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"display": {"interface": "cli"}},
    )

    main_mod._prepare_agent_startup(_args(command="chat"))

    assert plugin_calls == []
    assert mcp_calls == []


def test_non_tty_beats_hermes_tui_env_and_keeps_cli_discovery(monkeypatch):
    plugin_calls = _install_discover_spy(monkeypatch)
    mcp_calls = _install_mcp_and_hook_spies(monkeypatch)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setenv("HERMES_TUI", "1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"display": {"interface": "cli"}},
    )

    main_mod._prepare_agent_startup(_args(command="chat"))

    assert plugin_calls == ["discover"]
    assert mcp_calls == ["discover"]


def test_plugin_discovery_runs_for_plain_chat(monkeypatch):
    calls = _install_discover_spy(monkeypatch)
    mcp_calls = _install_mcp_and_hook_spies(monkeypatch)
    main_mod._prepare_agent_startup(_args(tui=False, command="chat"))
    assert calls == ["discover"]
    assert mcp_calls == ["discover"]
