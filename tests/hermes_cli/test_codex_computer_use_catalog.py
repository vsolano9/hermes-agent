"""Catalog contract for the signed local Codex Computer Use MCP."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest
import yaml

from hermes_cli import mcp_catalog


EXPECTED_TOOLS = [
    "list_apps",
    "get_app_state",
    "click",
    "perform_secondary_action",
    "set_value",
    "select_text",
    "scroll",
    "drag",
    "press_key",
    "type_text",
]


def _entry():
    entry = mcp_catalog.get_entry("codex-computer-use")
    assert entry is not None
    return entry


def test_catalog_entry_builds_local_verified_serial_mcp_config():
    entry = _entry()

    config = mcp_catalog._build_server_config(entry, install_dir=None)

    assert config == {
        "command": str((entry.manifest_path.parent / "launcher.py").resolve()),
        "trust": "untrusted",
        "supports_parallel_tool_calls": False,
        "single_writer": True,
        "minimal_env": True,
        "compatibility": {
            "tools_sha256": "dd485a140f5fbebe14147fb3ee2ed3914618b3484964efe02262b2479b322f1d",
            "capabilities_sha256": "52aa21370a62916d63adb5718fa1be519ec0fe4390136bf36e701be54e5582a5",
            "tool_count": 10,
            "tools_only": True,
        },
        "tools": {
            "include": EXPECTED_TOOLS,
            "resources": False,
            "prompts": False,
        },
    }


def test_catalog_entry_pins_exact_cua_tool_surface_without_probe(monkeypatch):
    entry = _entry()
    writes = []
    monkeypatch.setattr(
        mcp_catalog,
        "_probe_tools",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("fixed tool surfaces must not probe or widen")
        ),
    )
    monkeypatch.setattr(
        mcp_catalog,
        "_write_fixed_tools_policy",
        lambda name, include: writes.append((name, include)),
    )

    mcp_catalog._apply_tool_selection(
        entry,
        prior_selection=["unexpected_future_tool"],
        prior_exclude=["click"],
    )

    assert writes == [("codex-computer-use", EXPECTED_TOOLS)]


def test_fixed_policy_disables_resource_and_prompt_utility_tools(monkeypatch):
    state = {
        "mcp_servers": {
            "codex-computer-use": {"command": "/verified/launcher.py"}
        }
    }
    monkeypatch.setattr(mcp_catalog, "load_config", lambda: state)
    monkeypatch.setattr(mcp_catalog, "save_config", lambda cfg: state.update(cfg))

    mcp_catalog._write_fixed_tools_policy(
        "codex-computer-use", EXPECTED_TOOLS
    )

    tools = state["mcp_servers"]["codex-computer-use"]["tools"]
    assert tools == {
        "include": EXPECTED_TOOLS,
        "resources": False,
        "prompts": False,
    }


def test_catalog_entry_is_local_only_and_does_not_redistribute_openai_assets():
    entry = _entry()
    shipped_files = {
        path.relative_to(entry.manifest_path.parent).as_posix()
        for path in entry.manifest_path.parent.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }

    assert shipped_files == {"launcher.py", "manifest.yaml"}
    assert entry.transport.type == "stdio"
    assert entry.install is None
    assert entry.auth.type == "none"


def _write_manifest(tmp_path: Path, **updates) -> Path:
    data = {
        "manifest_version": 1,
        "name": "test-entry",
        "description": "Test entry",
        "source": "https://example.invalid",
        "transport": {"type": "stdio", "command": "local-client"},
        "auth": {"type": "none"},
    }
    data.update(updates)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("field", ["single_writer", "supports_parallel_tool_calls"])
def test_catalog_rejects_non_boolean_transport_safety_flags(tmp_path, field):
    path = _write_manifest(
        tmp_path,
        transport={"type": "stdio", "command": "local-client", field: "false"},
    )

    with pytest.raises(mcp_catalog.CatalogError, match=field):
        mcp_catalog._parse_manifest(path)


def test_catalog_rejects_widenable_or_malformed_fixed_policy(tmp_path):
    mixed = _write_manifest(
        tmp_path,
        tools={"fixed_enabled": ["click"], "default_enabled": ["future-tool"]},
    )
    with pytest.raises(mcp_catalog.CatalogError, match="mutually exclusive"):
        mcp_catalog._parse_manifest(mixed)

    invalid = _write_manifest(tmp_path, tools={"fixed_enabled": ["click", ""]})
    with pytest.raises(mcp_catalog.CatalogError, match="fixed_enabled"):
        mcp_catalog._parse_manifest(invalid)


def test_catalog_rejects_unknown_trust_tier(tmp_path):
    path = _write_manifest(tmp_path, trust="signed-ish")

    with pytest.raises(mcp_catalog.CatalogError, match="trust"):
        mcp_catalog._parse_manifest(path)


@pytest.mark.macos_only
def test_live_launcher_initialize_list_matches_pinned_tools_only_surface():
    if sys.platform != "darwin":
        pytest.skip("installed local Codex CUA exists only on macOS")
    entry = _entry()
    config = mcp_catalog._build_server_config(entry, install_dir=None)
    launcher = Path(config["command"])
    if not launcher.exists():
        pytest.skip("local launcher is unavailable")
    spec = importlib.util.spec_from_file_location("codex_cua_live_launcher", launcher)
    assert spec is not None and spec.loader is not None
    launcher_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher_module)
    try:
        installation = launcher_module.resolve_installation()
    except OSError:
        pytest.skip("installed Codex CUA dependency is unavailable")
    if not (
        installation.service_bundle.is_dir()
        and installation.client_bundle.is_dir()
        and installation.executable.is_file()
    ):
        pytest.skip("installed Codex CUA dependency is unavailable")

    from tools import mcp_tool

    mcp_tool._ensure_mcp_loop()

    async def inspect():
        server = await mcp_tool._connect_server(entry.name, config)
        try:
            capabilities = server.initialize_result.capabilities
            mcp_tool._validate_mcp_surface(
                entry.name, server._tools, capabilities, config
            )
            return [tool.name for tool in server._tools], capabilities
        finally:
            await server.shutdown()

    try:
        names, capabilities = mcp_tool._run_on_mcp_loop(inspect(), timeout=30)
    finally:
        mcp_tool._stop_mcp_loop_if_idle()

    assert names == EXPECTED_TOOLS
    assert (
        mcp_tool._mcp_capabilities_sha256(capabilities)
        == config["compatibility"]["capabilities_sha256"]
    )
    assert capabilities.resources is None
    assert capabilities.prompts is None
