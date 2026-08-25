"""Behavior-contract compatibility tests for native Hermes plugins."""

from pathlib import Path
import shutil

import yaml

from hermes_cli.plugins import PluginManager
from tools.registry import registry


LEGACY_PLUGIN = Path(__file__).parent / "fixtures" / "plugin_compat_legacy"


def test_legacy_plugin_loads_and_ignores_additive_hook_and_manifest_fields(
    tmp_path, monkeypatch
):
    """A frozen plugin keeps working as manifests and hook payloads grow."""
    hermes_home = tmp_path / "hermes-home"
    plugins_dir = hermes_home / "plugins"
    plugins_dir.mkdir(parents=True)
    shutil.copytree(LEGACY_PLUGIN, plugins_dir / "legacy-contract-fixture")
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"plugins": {"enabled": ["legacy-contract-fixture"]}}
        ),
        encoding="utf-8",
    )
    empty_bundled = tmp_path / "bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["legacy-contract-fixture"]
    assert loaded.enabled is True
    assert loaded.error is None
    assert loaded.module is not None
    assert loaded.manifest.version == "0.1.0"
    assert "on_session_start" in loaded.hooks_registered

    results = manager.invoke_hook(
        "on_session_start",
        session_id="legacy-session",
        resumed=True,
        future_additive_field={"nested": "value"},
    )

    assert results == [{"legacy_session_id": "legacy-session"}]
    assert loaded.module.received_sessions == ["legacy-session"]


def test_discovered_legacy_tool_handler_receives_no_invocation_keyword(
    tmp_path, monkeypatch
):
    """A frozen ``handler(args)`` remains callable through discovery unchanged."""
    hermes_home = tmp_path / "hermes-home"
    plugin_dir = hermes_home / "plugins" / "legacy-tool"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "legacy-tool", "version": "0.1.0"}),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "calls = []\n"
        "def handler(args):\n"
        "    calls.append(args)\n"
        "    return 'legacy-result'\n"
        "def register(ctx):\n"
        "    ctx.register_tool(\n"
        "        name='legacy_handler_probe', toolset='plugin_legacy_tool',\n"
        "        schema={'name': 'legacy_handler_probe', 'description': 'legacy', "
        "'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=handler, is_async=False)\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["legacy-tool"]}}),
        encoding="utf-8",
    )
    empty_bundled = tmp_path / "bundled-plugins"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))

    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["legacy-tool"]
    try:
        assert registry.dispatch(
            "legacy_handler_probe", {"unchanged": True}, scope=manager.scope_key
        ) == "legacy-result"
        assert loaded.module.calls == [{"unchanged": True}]
        entry = registry.get_entry("legacy_handler_probe", scope=manager.scope_key)
        assert entry.handler is loaded.module.handler
        assert entry.is_async is False
    finally:
        manager.unload("legacy-tool")
