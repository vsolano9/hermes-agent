"""Host-pinned MCP surfaces defer every upstream connection until dispatch."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tools import mcp_tool
from tools.registry import ToolRegistry


TOOLS_SHA256 = "dd485a140f5fbebe14147fb3ee2ed3914618b3484964efe02262b2479b322f1d"
CAPABILITIES_SHA256 = "52aa21370a62916d63adb5718fa1be519ec0fe4390136bf36e701be54e5582a5"
RAW_TOOLS = [
    "list_apps", "get_app_state", "click", "perform_secondary_action",
    "set_value", "select_text", "scroll", "drag", "press_key", "type_text",
]


def _config() -> dict:
    return {
        "transport": "codex_app_server",
        "single_writer": True,
        "trust": "untrusted",
        "minimal_env": True,
        "supports_parallel_tool_calls": False,
        "compatibility": {
            "app_server_catalog_sha256": "bc4f6aca3e12fecaa3d7eee6c800f7885790c3cdebe27b84e2e1ca5d3a020c38",
            "tools_sha256": TOOLS_SHA256,
            "capabilities_sha256": CAPABILITIES_SHA256,
            "tool_count": 10,
            "tools_only": True,
        },
        "tools": {
            "include": list(RAW_TOOLS),
            "resources": False,
            "prompts": False,
        },
    }


def _submit_invalid_policy_reregistration(alias: str) -> None:
    invalid = _config()
    invalid["trust"] = "full"
    mcp_tool.register_mcp_servers({alias: invalid})


def _clear_state() -> None:
    # These tests intentionally exercise the process-global registry used by
    # production discovery.  Restore every pinned publication they created so
    # later test modules do not inherit model-visible handlers or aliases.
    from tools.registry import registry

    pinned_servers = set(mcp_tool._pinned_lazy_server_configs)
    pinned_servers.update(mcp_tool._pinned_lazy_server_tool_names)
    pinned_toolsets = {f"mcp-{name}" for name in pinned_servers}
    with registry.registration_transaction():
        registry._tools = {
            tool_name: entry
            for tool_name, entry in registry._tools.items()
            if entry.toolset not in pinned_toolsets
        }
        for scope, entries in list(registry._scoped_tools.items()):
            kept = {
                tool_name: entry
                for tool_name, entry in entries.items()
                if entry.toolset not in pinned_toolsets
            }
            if kept:
                registry._scoped_tools[scope] = kept
            else:
                registry._scoped_tools.pop(scope, None)
        for alias, target in list(registry._toolset_aliases.items()):
            if alias in pinned_servers or target in pinned_toolsets:
                registry._toolset_aliases.pop(alias, None)
    for server_name in pinned_servers:
        mcp_tool._server_trust_levels.pop(server_name, None)
        mcp_tool._tool_read_only_hints.pop(server_name, None)
        mcp_tool._mcp_server_capability_identities.pop(server_name, None)
        mcp_tool._parallel_safe_servers.discard(server_name)
        mcp_tool._server_connect_errors.pop(server_name, None)
    for tool_name, server_name in list(mcp_tool._mcp_tool_server_names.items()):
        if server_name in pinned_servers:
            mcp_tool._mcp_tool_server_names.pop(tool_name, None)
    mcp_tool._pinned_lazy_server_configs.clear()
    mcp_tool._pinned_lazy_server_tool_names.clear()
    mcp_tool._servers.clear()
    mcp_tool._server_connecting.clear()
    mcp_tool._reset_single_writer_leases_for_tests()


def setup_function() -> None:
    _clear_state()


def teardown_function() -> None:
    _clear_state()


def test_checked_in_surface_reproduces_live_fingerprints() -> None:
    surface = mcp_tool._get_host_pinned_surface(_config())
    assert surface is not None
    assert [tool.name for tool in surface.tools] == RAW_TOOLS
    assert mcp_tool._mcp_surface_sha256(surface.tools) == TOOLS_SHA256
    assert mcp_tool._mcp_capabilities_sha256(surface.capabilities) == CAPABILITIES_SHA256

    poisoned = _config()
    poisoned["compatibility"] = dict(poisoned["compatibility"])
    poisoned["compatibility"]["tools_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="literal pinned digest"):
        mcp_tool._get_host_pinned_surface(poisoned)

    wrong_adapter = _config()
    wrong_adapter["transport"] = "stdio"
    with pytest.raises(ValueError, match="host adapter"):
        mcp_tool._get_host_pinned_surface(wrong_adapter)


@pytest.mark.parametrize(
    "mutation",
    ["missing_compatibility", "changed_digest", "normalized_trust", "missing_allowlist"],
)
def test_canonical_adapter_policy_drift_is_rejected_without_eager_fallback(
    mutation: str,
) -> None:
    config = _config()
    if mutation == "missing_compatibility":
        config.pop("compatibility")
    elif mutation == "changed_digest":
        config["compatibility"] = dict(config["compatibility"])
        config["compatibility"]["tools_sha256"] = "0" * 64
    elif mutation == "normalized_trust":
        config["trust"] = " UNTRUSTED "
    else:
        config.pop("tools")

    registry = ToolRegistry()
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._run_on_mcp_loop") as upstream:
        names = mcp_tool.register_mcp_servers({"renamed-cua": config})

    assert names == []
    upstream.assert_not_called()
    assert "renamed-cua" in mcp_tool._server_connect_errors
    assert "renamed-cua" not in mcp_tool._pinned_lazy_server_configs


def test_noncanonical_adapter_cannot_claim_cua_identity_from_digest_pair() -> None:
    config = _config()
    config["transport"] = "stdio"

    assert mcp_tool._single_writer_capability_key("digest-impostor", config) != (
        "openai-codex-cua"
    )


def test_canonical_adapter_aliases_share_one_capability_lock(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(mcp_tool, "_machine_account_home_for_lock", lambda: tmp_path)
    for alias in ("first-cua-alias", "second-cua-alias"):
        mcp_tool._configure_single_writer_server(
            alias, enabled=True, wait_timeout_seconds=0,
            capability_key=mcp_tool._single_writer_capability_key(alias, _config()),
        )

    assert mcp_tool._acquire_single_writer_lease(
        "first-cua-alias", "turn-a", wait_timeout_seconds=0,
    )
    assert not mcp_tool._acquire_single_writer_lease(
        "second-cua-alias", "turn-b", wait_timeout_seconds=0,
    )


def test_static_registration_never_uses_writable_schema_cache() -> None:
    registry = ToolRegistry()
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_schema_cache.get_cached_entry") as cache_read, \
         patch("tools.mcp_schema_cache.write_cache_entry") as cache_write, \
         patch("tools.mcp_tool._ensure_mcp_loop") as loop:
        names = mcp_tool.register_mcp_servers(
            {"codex-computer-use": _config()}
        )

    assert len(names) == 10
    cache_read.assert_not_called()
    cache_write.assert_not_called()
    loop.assert_not_called()


def test_static_publication_is_model_visible_without_upstream_connection() -> None:
    registry = ToolRegistry()
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._ensure_mcp_loop") as loop, \
         patch("tools.mcp_tool._run_on_mcp_loop") as upstream:
        names = mcp_tool.register_mcp_servers(
            {"codex-computer-use": _config()}
        )
        definitions = registry.get_definitions(set(names), quiet=True)

    assert len(names) == 10
    assert {item["function"]["name"] for item in definitions} == set(names)
    assert mcp_tool._servers == {}
    loop.assert_not_called()
    upstream.assert_not_called()


def test_shutdown_and_reload_republish_exact_model_visible_surface() -> None:
    registry = ToolRegistry()
    alias = "reloadable-cua"
    toolset = f"mcp-{alias}"
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._ensure_mcp_loop") as loop, \
         patch("tools.mcp_tool._run_on_mcp_loop") as upstream:
        first_names = mcp_tool.register_mcp_servers({alias: _config()})
        assert len(registry.get_definitions(set(first_names), quiet=True)) == 10

        mcp_tool.shutdown_mcp_servers()

        assert registry.get_tool_names_for_toolset(toolset) == []
        assert registry.get_toolset_alias_target(alias) is None
        assert alias not in mcp_tool._pinned_lazy_server_configs
        assert alias not in mcp_tool._pinned_lazy_server_tool_names
        assert alias not in mcp_tool._server_trust_levels
        assert alias not in mcp_tool._tool_read_only_hints
        assert alias not in mcp_tool._mcp_server_capability_identities
        assert alias not in mcp_tool._parallel_safe_servers
        assert alias not in mcp_tool._single_writer_policies
        assert alias not in mcp_tool._single_writer_capability_keys
        assert alias not in mcp_tool._single_writer_leases
        assert alias not in mcp_tool._single_writer_process_locks
        assert not any(
            server_name == alias
            for server_name in mcp_tool._mcp_tool_server_names.values()
        )

        second_names = mcp_tool.register_mcp_servers({alias: _config()})
        second_definitions = registry.get_definitions(
            set(second_names), quiet=True
        )

    assert len(second_names) == 10
    assert {item["function"]["name"] for item in second_definitions} == set(
        second_names
    )
    assert mcp_tool._single_writer_policies[alias] == (90.0, 10.0)
    assert mcp_tool._single_writer_capability_keys[alias] == (
        mcp_tool._CODEX_CUA_CAPABILITY_IDENTITY
    )
    assert mcp_tool._servers == {}
    loop.assert_not_called()
    upstream.assert_not_called()


@pytest.mark.parametrize("reload_mode", ["removed", "disabled"])
def test_shutdown_reload_without_enabled_pinned_server_leaves_no_surface(
    reload_mode: str,
) -> None:
    registry = ToolRegistry()
    alias = "reload-disabled-cua"
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._ensure_mcp_loop") as loop, \
         patch("tools.mcp_tool._run_on_mcp_loop") as upstream:
        names = mcp_tool.register_mcp_servers({alias: _config()})
        assert len(names) == 10
        mcp_tool.shutdown_mcp_servers()

        if reload_mode == "disabled":
            disabled = _config()
            disabled["enabled"] = False
            assert mcp_tool.register_mcp_servers({alias: disabled}) == []

        assert registry.get_tool_names_for_toolset(f"mcp-{alias}") == []
        assert registry.get_toolset_alias_target(alias) is None
        assert alias not in mcp_tool._pinned_lazy_server_configs
        assert alias not in mcp_tool._pinned_lazy_server_tool_names
        assert not any(
            server_name == alias
            for server_name in mcp_tool._mcp_tool_server_names.values()
        )
        assert mcp_tool._servers == {}

    loop.assert_not_called()
    upstream.assert_not_called()


def test_shutdown_publication_removal_is_atomic_to_registry_readers() -> None:
    registry = ToolRegistry()
    alias = "atomic-shutdown-cua"
    first_removal = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    observed = []
    errors = []
    class PausingTools(dict):
        def pop(self, tool_name, default=None):
            result = super().pop(tool_name, default)
            if not first_removal.is_set():
                first_removal.set()
                assert reader_started.wait(timeout=5)
                reader_finished.wait(timeout=0.25)
            return result

    def shutdown() -> None:
        try:
            mcp_tool.shutdown_mcp_servers()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def read() -> None:
        try:
            assert first_removal.wait(timeout=5)
            reader_started.set()
            observed.append((
                len(registry.get_tool_names_for_toolset(f"mcp-{alias}")),
                registry.get_toolset_alias_target(alias),
            ))
            reader_finished.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        assert len(mcp_tool.register_mcp_servers({alias: _config()})) == 10
        registry._tools = PausingTools(registry._tools)
        shutdown_thread = threading.Thread(target=shutdown)
        reader_thread = threading.Thread(target=read)
        shutdown_thread.start()
        reader_thread.start()
        shutdown_thread.join(timeout=10)
        reader_thread.join(timeout=10)

    assert not shutdown_thread.is_alive()
    assert not reader_thread.is_alive()
    assert errors == []
    assert observed == [(0, None)]


@pytest.mark.parametrize(
    "mutation",
    [
        "alias", "handler", "schema", "trust", "hints", "identity", "provenance",
        "policy", "key",
    ],
)
def test_static_availability_fails_closed_on_companion_state_drift(
    mutation: str,
) -> None:
    registry = ToolRegistry()
    alias = "availability-drift-cua"
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})
        assert len(registry.get_definitions(set(names), quiet=True)) == 10

        if mutation == "alias":
            with registry.registration_transaction():
                registry._toolset_aliases.pop(alias, None)
        elif mutation == "handler":
            registry.get_entry(names[0]).handler = lambda _args: "replaced"
        elif mutation == "schema":
            registry.get_entry(names[0]).schema["description"] = "injected"
        elif mutation == "trust":
            mcp_tool._server_trust_levels[alias] = "full"
        elif mutation == "hints":
            mcp_tool._tool_read_only_hints[alias] = {}
        elif mutation == "identity":
            mcp_tool._mcp_server_capability_identities[alias] = "ordinary"
        elif mutation == "provenance":
            mcp_tool._mcp_tool_server_names.pop(names[0], None)
        elif mutation == "policy":
            mcp_tool._single_writer_policies.pop(alias, None)
        else:
            mcp_tool._single_writer_capability_keys.pop(alias, None)

        assert registry.get_definitions(set(names), quiet=True) == []


def test_connected_pinned_availability_still_requires_exact_policy() -> None:
    registry = ToolRegistry()
    alias = "connected-availability-cua"
    connected = SimpleNamespace(
        session=object(), _is_recycled_stdio=lambda: False,
    )
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})
        mcp_tool._servers[alias] = connected
        assert len(registry.get_definitions(set(names), quiet=True)) == 10

        mcp_tool._single_writer_policies.pop(alias, None)

        assert registry.get_definitions(set(names), quiet=True) == []


def test_shutdown_gate_blocks_visibility_and_new_pinned_connection() -> None:
    registry = ToolRegistry()
    alias = "shutdown-gate-cua"
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})
        mcp_tool._mcp_shutting_down = True
        try:
            assert registry.get_definitions(set(names), quiet=True) == []
            assert not mcp_tool._ensure_pinned_server_connected(alias)
            assert mcp_tool._get_connected_server_for_call(alias) is None
        finally:
            mcp_tool._mcp_shutting_down = False


def test_shutdown_cleanup_failure_rolls_back_publication_without_stale_lease() -> None:
    registry = ToolRegistry()
    alias = "rollback-shutdown-cua"
    calls = 0
    releases = []

    class Cookie:
        def release(self) -> None:
            releases.append("released")

    class FailingTools(dict):
        def pop(self, tool_name, default=None):
            nonlocal calls
            calls += 1
            result = super().pop(tool_name, default)
            if calls == 2:
                raise RuntimeError("injected shutdown cleanup failure")
            return result

    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})
        mcp_tool._single_writer_leases[alias] = ("stale-owner", 1.0)
        mcp_tool._single_writer_process_locks[alias] = (
            "stale-owner", Cookie(),
        )
        registry._tools = FailingTools(registry._tools)
        with pytest.raises(RuntimeError, match="injected shutdown cleanup failure"):
            mcp_tool.shutdown_mcp_servers()

        assert registry.get_tool_names_for_toolset(f"mcp-{alias}") == sorted(
            names
        )
        assert registry.get_toolset_alias_target(alias) == f"mcp-{alias}"
        assert len(registry.get_definitions(set(names), quiet=True)) == 10
        assert mcp_tool._single_writer_policies[alias] == (90.0, 10.0)
        assert mcp_tool._single_writer_capability_keys[alias] == (
            mcp_tool._CODEX_CUA_CAPABILITY_IDENTITY
        )
        assert alias not in mcp_tool._single_writer_leases
        assert alias not in mcp_tool._single_writer_process_locks
        assert releases == ["released"]


def test_shutdown_uses_canonical_names_despite_stale_ledger_and_scope() -> None:
    registry = ToolRegistry()
    alias = "stale-ledger-cua"
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})
        hidden_name = names[0]
        mcp_tool._pinned_lazy_server_tool_names[alias].remove(hidden_name)
        registry._scoped_tools[registry.current_scope_key()] = {
            hidden_name: registry.get_entry(hidden_name)
        }

        mcp_tool.shutdown_mcp_servers()

        assert hidden_name not in registry._tools
        assert not any(
            hidden_name in entries
            for entries in registry._scoped_tools.values()
        )
        assert hidden_name not in mcp_tool._mcp_tool_server_names
        assert registry.get_toolset_alias_target(alias) is None


def test_shutdown_recovers_exact_surface_with_all_companion_maps_lost() -> None:
    registry = ToolRegistry()
    alias = "registry-only-cua"
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})
        mcp_tool._pinned_lazy_server_configs.clear()
        mcp_tool._pinned_lazy_server_tool_names.clear()
        mcp_tool._server_trust_levels.clear()
        mcp_tool._tool_read_only_hints.clear()
        mcp_tool._mcp_server_capability_identities.clear()
        mcp_tool._mcp_tool_server_names.clear()
        mcp_tool._single_writer_policies.clear()
        mcp_tool._single_writer_capability_keys.clear()

        mcp_tool.shutdown_mcp_servers()

        assert not any(registry.snapshot_registration(name) for name in names)
        assert registry.get_toolset_alias_target(alias) is None


def test_shutdown_does_not_misclassify_ordinary_cua_like_tool_name() -> None:
    registry = ToolRegistry()
    name = "mcp__ordinary__click"
    registry.register(
        name=name,
        toolset="mcp-ordinary",
        schema={"name": name, "description": "ordinary", "parameters": {}},
        handler=lambda _args: "ordinary",
    )
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        mcp_tool.shutdown_mcp_servers()

    assert registry.snapshot_registration(name) is not None


def test_static_publication_collision_rolls_back_every_new_handler() -> None:
    registry = ToolRegistry()
    occupied = "mcp__renamed_cua__click"
    registry.register(
        name=occupied,
        toolset="builtin-owner",
        schema={"name": occupied, "description": "occupied", "parameters": {}},
        handler=lambda _args: "occupied",
        check_fn=lambda: True,
        is_async=False,
        description="occupied",
    )
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({"renamed-cua": _config()})

    assert names == []
    assert registry.get_toolset_for_tool(occupied) == "builtin-owner"
    assert registry.get_tool_names_for_toolset("mcp-renamed-cua") == []
    assert "renamed-cua" not in mcp_tool._pinned_lazy_server_configs
    assert "renamed-cua" not in mcp_tool._pinned_lazy_server_tool_names


def test_static_publication_registry_rejection_rolls_back_prior_handlers() -> None:
    registry = ToolRegistry()
    real_register = registry.register

    def reject_last(*args, **kwargs):
        name = kwargs.get("name") or args[0]
        if name.endswith("__type_text"):
            return None
        return real_register(*args, **kwargs)

    with patch("tools.registry.registry", registry), \
         patch.object(registry, "register", side_effect=reject_last), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({"renamed-cua": _config()})

    assert names == []
    assert registry.get_tool_names_for_toolset("mcp-renamed-cua") == []
    assert "renamed-cua" not in mcp_tool._pinned_lazy_server_configs
    assert "renamed-cua" not in mcp_tool._pinned_lazy_server_tool_names


def test_static_publication_is_atomic_to_concurrent_registry_reader() -> None:
    registry = ToolRegistry()
    real_register = registry.register
    first_handler_published = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    observed_states = []
    published_names = []
    errors = []

    def pause_after_first_handler(*args, **kwargs):
        result = real_register(*args, **kwargs)
        if not first_handler_published.is_set():
            first_handler_published.set()
            assert reader_started.wait(timeout=5)
            # An atomic publisher keeps the reader blocked here until the
            # complete ten-tool set and its alias are visible.
            reader_finished.wait(timeout=0.25)
        return result

    def publish() -> None:
        try:
            with patch("tools.registry.registry", registry), \
                 patch.object(registry, "register", side_effect=pause_after_first_handler), \
                 patch("tools.mcp_tool._MCP_AVAILABLE", True):
                published_names.extend(
                    mcp_tool.register_mcp_servers({"renamed-cua": _config()})
                )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def read() -> None:
        try:
            assert first_handler_published.wait(timeout=5)
            reader_started.set()
            observed_states.append((
                len(registry.get_tool_names_for_toolset("mcp-renamed-cua")),
                registry.get_toolset_alias_target("renamed-cua"),
            ))
            reader_finished.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    publisher = threading.Thread(target=publish)
    reader = threading.Thread(target=read)
    publisher.start()
    reader.start()
    publisher.join(timeout=10)
    reader.join(timeout=10)

    assert not publisher.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert len(published_names) == 10
    assert observed_states == [(10, "mcp-renamed-cua")]


def test_static_publication_does_not_expose_handlers_before_untrusted_policy() -> None:
    class BlockingTrustMap(dict):
        def __setitem__(self, key, value):
            if key == "atomic-policy-cua":
                handlers_visible.set()
                assert reader_started.wait(timeout=5)
                # With complete publication atomicity, the registry reader
                # remains blocked until this untrusted policy is installed.
                reader_finished.wait(timeout=0.25)
            return super().__setitem__(key, value)

    registry = ToolRegistry()
    handlers_visible = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    gate_results = []
    errors = []
    trust_map = BlockingTrustMap(mcp_tool._server_trust_levels)

    def publish() -> None:
        try:
            mcp_tool.register_mcp_servers({"atomic-policy-cua": _config()})
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def read_and_gate() -> None:
        try:
            assert handlers_visible.wait(timeout=5)
            reader_started.set()
            entry = registry.get_entry("mcp__atomic_policy_cua__click")
            assert entry is not None
            gate_results.append(mcp_tool._trust_gate_check(
                "atomic-policy-cua",
                "click",
                {"app": "TextEdit", "element_id": 1},
                "concurrent-reader",
            ))
            reader_finished.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with patch("tools.registry.registry", registry), \
         patch.object(mcp_tool, "_server_trust_levels", trust_map), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch(
             "tools.approval.request_elicitation_consent",
             lambda *_args, **_kwargs: "deny",
         ):
        publisher = threading.Thread(target=publish)
        reader = threading.Thread(target=read_and_gate)
        publisher.start()
        reader.start()
        publisher.join(timeout=10)
        reader.join(timeout=10)

    assert not publisher.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert len(gate_results) == 1
    assert gate_results[0] is not None


@pytest.mark.parametrize(
    ("map_name", "failure_index"),
    [
        ("_server_trust_levels", 1),
        ("_tool_read_only_hints", 1),
        ("_mcp_tool_server_names", 2),
        ("_pinned_lazy_server_configs", 1),
        ("_pinned_lazy_server_tool_names", 1),
        ("_single_writer_policies", 1),
        ("_single_writer_capability_keys", 1),
    ],
)
def test_static_publication_external_write_failure_has_zero_residue(
    map_name: str, failure_index: int,
) -> None:
    class FailOnceDict(dict):
        def __init__(self, initial):
            super().__init__(initial)
            self.writes = 0
            self.failed = False

        def __setitem__(self, key, value):
            self.writes += 1
            super().__setitem__(key, value)
            if not self.failed and self.writes == failure_index:
                self.failed = True
                raise RuntimeError(f"injected {map_name} write failure")

    registry = ToolRegistry()
    alias = f"failure-{map_name.strip('_').replace('_', '-')}-cua"
    expected_names = {
        mcp_tool.mcp_prefixed_tool_name(alias, raw_name)
        for raw_name in RAW_TOOLS
    }
    failing_map = FailOnceDict(getattr(mcp_tool, map_name))

    with patch("tools.registry.registry", registry), \
         patch.object(mcp_tool, map_name, failing_map), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})

        assert names == []
        assert registry.get_tool_names_for_toolset(f"mcp-{alias}") == []
        assert registry.get_toolset_alias_target(alias) is None
        assert alias not in mcp_tool._server_trust_levels
        assert alias not in mcp_tool._tool_read_only_hints
        assert alias not in mcp_tool._pinned_lazy_server_configs
        assert alias not in mcp_tool._pinned_lazy_server_tool_names
        assert alias not in mcp_tool._mcp_server_capability_identities
        assert alias not in mcp_tool._parallel_safe_servers
        assert alias not in mcp_tool._single_writer_policies
        assert alias not in mcp_tool._single_writer_capability_keys
        assert alias not in mcp_tool._single_writer_leases
        assert alias not in mcp_tool._single_writer_process_locks
        assert expected_names.isdisjoint(mcp_tool._mcp_tool_server_names)


def test_static_publication_parallel_policy_failure_rolls_back_everything() -> None:
    class FailAfterDiscard(set):
        def __init__(self, initial, fail_name):
            super().__init__(initial)
            self.fail_name = fail_name
            self.failed = False

        def discard(self, value):
            super().discard(value)
            if value == self.fail_name and not self.failed:
                self.failed = True
                raise RuntimeError("injected parallel policy write failure")

    registry = ToolRegistry()
    alias = "failure-parallel-policy-cua"
    parallel = FailAfterDiscard(
        set(mcp_tool._parallel_safe_servers) | {alias}, alias
    )
    with patch("tools.registry.registry", registry), \
         patch.object(mcp_tool, "_parallel_safe_servers", parallel), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})

    assert names == []
    assert alias in parallel
    assert registry.get_tool_names_for_toolset(f"mcp-{alias}") == []
    assert alias not in mcp_tool._server_trust_levels
    assert alias not in mcp_tool._mcp_server_capability_identities
    assert alias not in mcp_tool._single_writer_policies
    assert alias not in mcp_tool._single_writer_capability_keys


def test_invalid_canonical_config_cannot_relabel_existing_ordinary_server() -> None:
    registry = ToolRegistry()
    alias = "existing-ordinary-server"
    ordinary = {
        "command": sys.executable,
        "args": ["-c", "raise SystemExit(0)"],
        "enabled": False,
    }
    invalid_canonical = _config()
    invalid_canonical["trust"] = "full"

    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        mcp_tool.register_mcp_servers({alias: ordinary})
        assert mcp_tool._mcp_server_capability_identities.get(alias) == (
            f"mcp-server:{alias}"
        )
        identity_state_before = dict(mcp_tool._mcp_server_capability_identities)

        mcp_tool.register_mcp_servers({alias: invalid_canonical})

    assert mcp_tool._mcp_server_capability_identities == identity_state_before


def test_reserved_name_ordinary_server_gets_no_cua_capabilities() -> None:
    registry = ToolRegistry()
    reserved_name = mcp_tool._CODEX_CUA_CAPABILITY_IDENTITY
    ordinary = {
        "command": sys.executable,
        "args": ["-c", "raise SystemExit(0)"],
        "enabled": False,
    }
    real_cua_alias = "real-cua-for-reserved-name-test"

    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        mcp_tool.register_mcp_servers({reserved_name: ordinary})
        mcp_tool._remember_mcp_capability_identity(real_cua_alias, _config())

    assert mcp_tool._mcp_capability_identity(reserved_name) == (
        f"mcp-server:{reserved_name}"
    )
    assert not mcp_tool._is_codex_cua_identity(reserved_name)
    assert mcp_tool._single_writer_capability_key(reserved_name, ordinary) == (
        f"mcp-server:{reserved_name}"
    )
    assert mcp_tool._single_writer_capability_key(reserved_name, ordinary) != (
        mcp_tool._single_writer_capability_key(real_cua_alias, _config())
    )

    raw_result = json.dumps({"app": "Safari", "windows": []})
    assert mcp_tool._record_state_result_for_exact_grant(
        server_name=reserved_name,
        tool_name="get_app_state",
        args={"app": "Safari"},
        task_id="reserved-name-child",
        result=raw_result,
    ) == raw_result

    mcp_tool._record_mcp_state_observation(
        task_id="real-child",
        server_name=real_cua_alias,
        app="Safari",
        state_digest="a" * 64,
    )
    mcp_tool._issue_mcp_exact_action_grant(
        task_id="real-child",
        server_name=real_cua_alias,
        app="Safari",
        state_digest="a" * 64,
        tool_name="click",
        arguments={"app": "Safari", "element_id": 1},
        ttl_seconds=30,
    )
    assert not mcp_tool._consume_mcp_exact_action_grant(
        reserved_name,
        "click",
        {"app": "Safari", "element_id": 1},
        "real-child",
    )


def test_concurrent_invalid_reregistration_keeps_handlers_and_policy() -> None:
    registry = ToolRegistry()
    alias = "policy-drift-race-cua"
    click_name = mcp_tool.mcp_prefixed_tool_name(alias, "click")
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        assert len(mcp_tool.register_mcp_servers({alias: _config()})) == 10

        validation_started = threading.Event()
        reader_started = threading.Event()
        reader_finished = threading.Event()
        observed = []
        errors = []
        real_classify = mcp_tool._get_host_pinned_surface

        def pause_classification(config):
            validation_started.set()
            assert reader_started.wait(timeout=5)
            reader_finished.wait(timeout=0.25)
            return real_classify(config)

        def reject_invalid() -> None:
            try:
                with patch(
                    "tools.mcp_tool._get_host_pinned_surface",
                    side_effect=pause_classification,
                ):
                    _submit_invalid_policy_reregistration(alias)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        def read() -> None:
            try:
                assert validation_started.wait(timeout=5)
                reader_started.set()
                entry = registry.get_entry(click_name)
                observed.append({
                    "handler": entry is not None,
                    "trust": mcp_tool._server_trust_levels.get(alias),
                    "pinned": alias in mcp_tool._pinned_lazy_server_configs,
                    "provenance": mcp_tool._mcp_tool_server_names.get(click_name),
                })
                reader_finished.set()
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        writer = threading.Thread(target=reject_invalid)
        reader = threading.Thread(target=read)
        writer.start()
        reader.start()
        writer.join(timeout=10)
        reader.join(timeout=10)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert observed == [{
        "handler": True,
        "trust": "untrusted",
        "pinned": True,
        "provenance": alias,
    }]


@pytest.mark.parametrize(
    "corrupt_ledger",
    [
        lambda names: names[:1],
        lambda names: names[:5],
        lambda _names: [],
        lambda names: [names[0], "not-a-registered-tool", None, 42],
    ],
    ids=["stale-one", "partial", "empty", "corrupt"],
)
def test_invalid_reregistration_preserves_existing_state_despite_bad_ledger(
    corrupt_ledger,
) -> None:
    registry = ToolRegistry()
    alias = "bad-ledger-cua"
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})
        assert len(names) == 10
        corrupted = corrupt_ledger(names)
        with mcp_tool._lock:
            mcp_tool._pinned_lazy_server_tool_names[alias] = corrupted

        _submit_invalid_policy_reregistration(alias)

        assert len(registry.get_tool_names_for_toolset(f"mcp-{alias}")) == 10
        assert registry.get_toolset_alias_target(alias) == f"mcp-{alias}"
        assert mcp_tool._server_trust_levels.get(alias) == "untrusted"
        assert alias in mcp_tool._tool_read_only_hints
        assert alias in mcp_tool._pinned_lazy_server_configs
        assert mcp_tool._pinned_lazy_server_tool_names.get(alias) == corrupted
        assert all(
            mcp_tool._mcp_tool_server_names.get(name) == alias
            for name in names
        )


def test_invalid_reregistration_preserves_global_and_scope_overlays() -> None:
    registry = ToolRegistry()
    alias = "scoped-hide-cua"
    toolset = f"mcp-{alias}"
    click_name = mcp_tool.mcp_prefixed_tool_name(alias, "click")
    type_name = mcp_tool.mcp_prefixed_tool_name(alias, "type_text")
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        assert len(mcp_tool.register_mcp_servers({alias: _config()})) == 10

        # Caller-controlled invalid input cannot delete either the healthy
        # global publication or registrations in any profile scope.
        registry.register(
            name=click_name,
            toolset="scoped-owner",
            schema={"name": click_name, "description": "overlay", "parameters": {}},
            handler=lambda _args: "overlay",
            override=True,
            scope="scope-a",
        )
        registry.register(
            name=type_name,
            toolset=toolset,
            schema={"name": type_name, "description": "pinned overlay", "parameters": {}},
            handler=lambda _args: "pinned-overlay",
            scope="scope-b",
        )

        with patch.object(registry, "current_scope_key", return_value="scope-a"):
            _submit_invalid_policy_reregistration(alias)

        assert registry.snapshot_registration(click_name) is not None
        assert registry.snapshot_registration(type_name) is not None
        scoped_owner = registry.snapshot_registration(click_name, scope="scope-a")
        assert scoped_owner is not None
        assert scoped_owner.toolset == "scoped-owner"
        assert registry.snapshot_registration(type_name, scope="scope-b") is not None
        assert registry.get_toolset_alias_target(alias) == toolset
        assert mcp_tool._server_trust_levels.get(alias) == "untrusted"
        assert alias in mcp_tool._pinned_lazy_server_configs
        assert alias in mcp_tool._mcp_tool_server_names.values()


def test_registry_exposes_no_cross_scope_bulk_delete_to_scoped_plugins() -> None:
    registry = ToolRegistry()
    name = "mcp__protected_global__click"
    registry.register(
        name=name,
        toolset="mcp-protected-global",
        schema={"name": name, "description": "global", "parameters": {}},
        handler=lambda _args: "global",
    )
    registry.register_plugin_override_policy(
        "hermes_plugins.attacker", False, scope="scope-a"
    )
    plugin_globals = {
        "__name__": "hermes_plugins.attacker.actions",
        "registry": registry,
    }
    exec(
        "def remove(tool_name):\n    registry.deregister(tool_name)",
        plugin_globals,
    )

    assert not hasattr(registry, "get_registration_slots_for_toolset")
    assert not hasattr(registry, "deregister_mcp_toolset_registrations")
    assert not hasattr(registry, "deregister_toolset_alias")
    assert not hasattr(mcp_tool, "_host_registry_slots_for_exact_toolset_locked")
    assert not hasattr(mcp_tool, "_host_deregister_exact_mcp_toolset_locked")
    assert not hasattr(mcp_tool, "_quarantine_host_pinned_publication")
    with patch.object(registry, "current_scope_key", return_value="scope-a"):
        with pytest.raises(PermissionError, match="process-global"):
            plugin_globals["remove"](name)

    assert registry.snapshot_registration(name) is not None


@pytest.mark.parametrize("hostile_kind", ["reserved-digests", "ordinary"])
def test_hostile_reregistration_preserves_existing_publication(
    hostile_kind: str, tmp_path: Path, monkeypatch,
) -> None:
    registry = ToolRegistry()
    alias = f"confused-deputy-{hostile_kind}-cua"
    toolset = f"mcp-{alias}"
    click_name = mcp_tool.mcp_prefixed_tool_name(alias, "click")
    validation_started = threading.Event()
    reader_started = threading.Event()
    reader_finished = threading.Event()
    observed_counts = []
    errors = []

    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({alias: _config()})
        assert len(names) == 10
        monkeypatch.setattr(
            mcp_tool, "_machine_account_home_for_lock", lambda: tmp_path
        )
        assert mcp_tool._acquire_single_writer_lease(
            alias, "active-owner", wait_timeout_seconds=0
        )
        registry.register(
            name=click_name,
            toolset="scoped-owner",
            schema={"name": click_name, "description": "overlay", "parameters": {}},
            handler=lambda _args: "overlay",
            override=True,
            scope="scope-a",
        )
        global_before = {
            name: registry.snapshot_registration(name)
            for name in names
        }
        scoped_before = registry.snapshot_registration(click_name, scope="scope-a")
        trust_before = dict(mcp_tool._server_trust_levels)
        hints_before = {
            key: dict(value)
            for key, value in mcp_tool._tool_read_only_hints.items()
        }
        provenance_before = dict(mcp_tool._mcp_tool_server_names)
        identities_before = dict(mcp_tool._mcp_server_capability_identities)
        configs_before = dict(mcp_tool._pinned_lazy_server_configs)
        names_before = {
            key: list(value)
            for key, value in mcp_tool._pinned_lazy_server_tool_names.items()
        }
        parallel_before = set(mcp_tool._parallel_safe_servers)
        writer_policies_before = dict(mcp_tool._single_writer_policies)
        writer_keys_before = dict(mcp_tool._single_writer_capability_keys)
        writer_leases_before = dict(mcp_tool._single_writer_leases)
        writer_locks_before = dict(mcp_tool._single_writer_process_locks)
        alias_before = registry.get_toolset_alias_target(alias)

        if hostile_kind == "reserved-digests":
            hostile = _config()
            hostile["command"] = "/usr/bin/false"
        else:
            hostile = {
                "command": "/usr/bin/false",
                "supports_parallel_tool_calls": True,
                "single_writer": False,
            }
        real_classify = mcp_tool._get_host_pinned_surface

        def pause_classification(config):
            validation_started.set()
            assert reader_started.wait(timeout=5)
            reader_finished.wait(timeout=0.25)
            return real_classify(config)

        def reregister() -> None:
            try:
                with patch(
                    "tools.mcp_tool._get_host_pinned_surface",
                    side_effect=pause_classification,
                ):
                    mcp_tool.register_mcp_servers({alias: hostile})
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        def read() -> None:
            try:
                assert validation_started.wait(timeout=5)
                reader_started.set()
                observed_counts.append(
                    len(registry.get_tool_names_for_toolset(toolset))
                )
                reader_finished.set()
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        writer = threading.Thread(target=reregister)
        reader = threading.Thread(target=read)
        writer.start()
        reader.start()
        writer.join(timeout=10)
        reader.join(timeout=10)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert observed_counts == [10]
    assert {
        name: registry.snapshot_registration(name)
        for name in names
    } == global_before
    assert registry.snapshot_registration(click_name, scope="scope-a") is scoped_before
    assert registry.get_toolset_alias_target(alias) == alias_before
    assert mcp_tool._server_trust_levels == trust_before
    assert mcp_tool._tool_read_only_hints == hints_before
    assert mcp_tool._mcp_tool_server_names == provenance_before
    assert mcp_tool._mcp_server_capability_identities == identities_before
    assert mcp_tool._pinned_lazy_server_configs == configs_before
    assert mcp_tool._pinned_lazy_server_tool_names == names_before
    assert mcp_tool._parallel_safe_servers == parallel_before
    assert mcp_tool._single_writer_policies == writer_policies_before
    assert mcp_tool._single_writer_capability_keys == writer_keys_before
    assert mcp_tool._single_writer_leases == writer_leases_before
    assert mcp_tool._single_writer_process_locks == writer_locks_before


def test_sequential_ordinary_duplicate_preserves_pinned_concurrency_policy() -> None:
    registry = ToolRegistry()
    alias = "sequential-duplicate-cua"
    hostile = {
        "command": "/usr/bin/false",
        "supports_parallel_tool_calls": True,
        "single_writer": False,
    }
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        assert len(mcp_tool.register_mcp_servers({alias: _config()})) == 10
        before = (
            set(mcp_tool._parallel_safe_servers),
            dict(mcp_tool._single_writer_policies),
            dict(mcp_tool._single_writer_capability_keys),
            dict(mcp_tool._mcp_server_capability_identities),
            dict(mcp_tool._server_trust_levels),
        )

        mcp_tool.register_mcp_servers({alias: hostile})

    assert (
        set(mcp_tool._parallel_safe_servers),
        dict(mcp_tool._single_writer_policies),
        dict(mcp_tool._single_writer_capability_keys),
        dict(mcp_tool._mcp_server_capability_identities),
        dict(mcp_tool._server_trust_levels),
    ) == before


def test_pinned_publication_wins_classify_to_policy_race(
    tmp_path: Path, monkeypatch,
) -> None:
    class BarrierRLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._counts = {}

        def acquire(self, *args, **kwargs):
            current = threading.current_thread().name
            count = self._counts.get(current, 0) + 1
            self._counts[current] = count
            if current == "ordinary-reregister" and count == 2:
                ordinary_reached_policy_boundary.set()
                assert pinned_committed.wait(timeout=5)
            return self._lock.acquire(*args, **kwargs)

        def release(self):
            return self._lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()

    registry = ToolRegistry()
    state_lock = BarrierRLock()
    alias = "classify-publish-race-cua"
    ordinary_reached_policy_boundary = threading.Event()
    pinned_committed = threading.Event()
    errors = []
    ordinary = {
        "command": "/usr/bin/false",
        "supports_parallel_tool_calls": True,
        "single_writer": False,
    }

    def register_ordinary() -> None:
        try:
            mcp_tool.register_mcp_servers({alias: ordinary})
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def publish_pinned() -> None:
        try:
            assert ordinary_reached_policy_boundary.wait(timeout=5)
            assert len(mcp_tool.register_mcp_servers({alias: _config()})) == 10
            pinned_committed.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
            pinned_committed.set()

    monkeypatch.setattr(mcp_tool, "_machine_account_home_for_lock", lambda: tmp_path)
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._lock", state_lock), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        ordinary_thread = threading.Thread(
            target=register_ordinary, name="ordinary-reregister"
        )
        pinned_thread = threading.Thread(
            target=publish_pinned, name="pinned-publisher"
        )
        ordinary_thread.start()
        pinned_thread.start()
        ordinary_thread.join(timeout=10)
        pinned_thread.join(timeout=10)

        assert not ordinary_thread.is_alive()
        assert not pinned_thread.is_alive()
        assert errors == []
        names = registry.get_tool_names_for_toolset(f"mcp-{alias}")
        assert len(names) == 10
        assert registry.get_toolset_alias_target(alias) == f"mcp-{alias}"
        assert mcp_tool._mcp_server_capability_identities.get(alias) == (
            mcp_tool._CODEX_CUA_CAPABILITY_IDENTITY
        )
        assert alias not in mcp_tool._parallel_safe_servers
        assert mcp_tool._single_writer_policies.get(alias) == (90.0, 10.0)
        assert mcp_tool._single_writer_capability_keys.get(alias) == (
            mcp_tool._CODEX_CUA_CAPABILITY_IDENTITY
        )
        assert mcp_tool._server_trust_levels.get(alias) == "untrusted"
        assert mcp_tool._pinned_lazy_server_configs.get(alias) == _config()
        assert all(
            mcp_tool._mcp_tool_server_names.get(name) == alias for name in names
        )


def test_pinned_publication_overrides_policy_committed_after_classification() -> None:
    registry = ToolRegistry()
    alias = "inverse-policy-race-cua"
    pinned_classified = threading.Event()
    ordinary_finished = threading.Event()
    errors = []
    ordinary = {
        "enabled": False,
        "command": "/usr/bin/false",
        "supports_parallel_tool_calls": True,
        "single_writer": False,
    }
    real_classify = mcp_tool._get_host_pinned_surface

    def pause_pinned_after_classification(config):
        surface = real_classify(config)
        if surface is not None:
            pinned_classified.set()
            assert ordinary_finished.wait(timeout=5)
        return surface

    def publish_pinned() -> None:
        try:
            with patch(
                "tools.mcp_tool._get_host_pinned_surface",
                side_effect=pause_pinned_after_classification,
            ):
                assert len(mcp_tool.register_mcp_servers({alias: _config()})) == 10
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        pinned_thread = threading.Thread(target=publish_pinned)
        pinned_thread.start()
        assert pinned_classified.wait(timeout=5)
        mcp_tool.register_mcp_servers({alias: ordinary})
        assert alias in mcp_tool._parallel_safe_servers
        assert alias not in mcp_tool._single_writer_policies
        ordinary_finished.set()
        pinned_thread.join(timeout=10)

    assert not pinned_thread.is_alive()
    assert errors == []
    assert len(registry.get_tool_names_for_toolset(f"mcp-{alias}")) == 10
    assert alias not in mcp_tool._parallel_safe_servers
    assert mcp_tool._single_writer_policies.get(alias) == (90.0, 10.0)
    assert mcp_tool._single_writer_capability_keys.get(alias) == (
        mcp_tool._CODEX_CUA_CAPABILITY_IDENTITY
    )
    assert mcp_tool._mcp_server_capability_identities.get(alias) == (
        mcp_tool._CODEX_CUA_CAPABILITY_IDENTITY
    )


def test_eight_processes_publish_ten_tools_without_connecting(tmp_path: Path) -> None:
    script = r'''
import json
from unittest.mock import patch
from tools import mcp_tool
cfg = json.loads(__import__('os').environ['PINNED_CONFIG'])
with patch('tools.mcp_tool._MCP_AVAILABLE', True), \
     patch('tools.mcp_tool._ensure_mcp_loop') as loop, \
     patch('tools.mcp_tool._run_on_mcp_loop') as run:
    names = mcp_tool.register_mcp_servers({'codex-computer-use': cfg})
print(json.dumps({'count': len(names), 'loop': loop.call_count, 'run': run.call_count}))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PINNED_CONFIG"] = json.dumps(_config())
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        for _ in range(8)
    ]
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        results.append(json.loads(stdout.strip().splitlines()[-1]))
    assert results == [{"count": 10, "loop": 0, "run": 0}] * 8


def test_eight_processes_serialize_the_upstream_lifetime(tmp_path: Path) -> None:
    account_home = tmp_path / "account"
    account_home.mkdir()
    state_path = tmp_path / "active.json"
    state_path.write_text('{"active":0,"maximum":0,"completed":0}')
    script = r'''
import fcntl, json, os, time
from pathlib import Path
from tools import mcp_tool
home = Path(os.environ['ACCOUNT_HOME'])
state_path = Path(os.environ['ACTIVE_STATE'])
mcp_tool._machine_account_home_for_lock = lambda: home
mcp_tool._configure_single_writer_server(
    'codex-computer-use', enabled=True, wait_timeout_seconds=10,
    capability_key='openai-codex-cua',
)
owner = 'turn-' + str(os.getpid())
assert mcp_tool._acquire_single_writer_lease(
    'codex-computer-use', owner, wait_timeout_seconds=10,
)
with state_path.open('r+') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    state = json.load(handle)
    state['active'] += 1
    state['maximum'] = max(state['maximum'], state['active'])
    handle.seek(0); json.dump(state, handle); handle.truncate()
    fcntl.flock(handle, fcntl.LOCK_UN)
time.sleep(0.03)
with state_path.open('r+') as handle:
    fcntl.flock(handle, fcntl.LOCK_EX)
    state = json.load(handle)
    state['active'] -= 1
    state['completed'] += 1
    handle.seek(0); json.dump(state, handle); handle.truncate()
    fcntl.flock(handle, fcntl.LOCK_UN)
mcp_tool.release_mcp_single_writer_leases(owner)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["ACCOUNT_HOME"] = str(account_home)
    env["ACTIVE_STATE"] = str(state_path)
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script], stderr=subprocess.PIPE, text=True,
            env=env,
        )
        for _ in range(8)
    ]
    for process in processes:
        _stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
    assert json.loads(state_path.read_text()) == {
        "active": 0, "maximum": 1, "completed": 8,
    }


def test_crashed_client_releases_account_global_capability(tmp_path: Path) -> None:
    account_home = tmp_path / "account"
    account_home.mkdir()
    common = r'''
from pathlib import Path
from tools import mcp_tool
mcp_tool._machine_account_home_for_lock = lambda: Path(%r)
mcp_tool._configure_single_writer_server(
    'codex-computer-use', enabled=True, wait_timeout_seconds=2,
    capability_key='openai-codex-cua',
)
''' % str(account_home)
    holder = subprocess.Popen(
        [sys.executable, "-c", common + r'''
assert mcp_tool._acquire_single_writer_lease('codex-computer-use', 'dead', wait_timeout_seconds=0)
print('ready', flush=True)
__import__('time').sleep(30)
'''],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    assert holder.stdout.readline().strip() == "ready"
    holder.kill()
    holder.wait(timeout=5)
    probe = subprocess.run(
        [sys.executable, "-c", common + r'''
assert mcp_tool._acquire_single_writer_lease('codex-computer-use', 'next', wait_timeout_seconds=2)
mcp_tool.release_mcp_single_writer_leases('next')
'''],
        capture_output=True, text=True, timeout=10,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    assert probe.returncode == 0, probe.stderr


def test_live_drift_fails_without_replacing_static_handlers() -> None:
    registry = ToolRegistry()
    config = _config()
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        names = mcp_tool.register_mcp_servers({"codex-computer-use": config})
        handlers = {name: registry.get_entry(name).handler for name in names}
        surface = mcp_tool._get_host_pinned_surface(config)
        drifted = list(surface.tools)
        drifted[0] = SimpleNamespace(**vars(drifted[0]))
        drifted[0].description = "injected replacement"
        server = SimpleNamespace(
            _tools=drifted,
            initialize_result=SimpleNamespace(capabilities=surface.capabilities),
            session=object(),
            shutdown=AsyncMock(),
        )

        async def connect(_name, _config, *, publish_tools=True):
            mcp_tool._validate_mcp_surface(
                "codex-computer-use", server._tools,
                server.initialize_result.capabilities, _config,
            )
            return []

        with patch("tools.mcp_tool._discover_and_register_server", side_effect=connect):
            assert not mcp_tool._ensure_pinned_server_connected("codex-computer-use")

        assert {name: registry.get_entry(name).handler for name in names} == handlers
        assert server.shutdown.await_count == 0


def test_sessionless_live_task_is_reused_without_second_discovery() -> None:
    name = "renamed-cua"
    server = mcp_tool.MCPServerTask(name)
    server._task = SimpleNamespace(done=lambda: False)
    mcp_tool._servers[name] = server
    mcp_tool._pinned_lazy_server_configs[name] = _config()

    def reconnect(*_args, **_kwargs):
        server.session = object()
        server._ready.set()
        return True

    with patch(
        "tools.mcp_tool._signal_reconnect_and_wait", side_effect=reconnect
    ) as signal, patch(
        "tools.mcp_tool._discover_and_register_server", new_callable=AsyncMock
    ) as discover:
        assert mcp_tool._ensure_pinned_server_connected(name)

    assert mcp_tool._servers[name] is server
    signal.assert_called_once()
    discover.assert_not_called()


def test_dispatch_acquires_writer_before_first_broker_call() -> None:
    order = []
    mcp_tool._pinned_lazy_server_configs["codex-computer-use"] = _config()
    handler = mcp_tool._make_tool_handler(
        "codex-computer-use", "list_apps", 5.0
    )
    broker = SimpleNamespace(call=lambda *_a, **_kw: (
        order.append("broker") or {
            "content": [{"type": "text", "text": "[]"}],
            "isError": False,
        }
    ))
    with patch("tools.mcp_tool._trust_gate_check", return_value=None), \
         patch(
             "tools.mcp_tool._acquire_single_writer_lease",
             side_effect=lambda *_a, **_kw: order.append("lease") or True,
         ), \
         patch("tools.mcp_tool._get_codex_cua_broker", return_value=broker), \
         patch(
             "tools.mcp_tool._get_connected_server_for_call",
             side_effect=AssertionError("broker must not create MCPServerTask"),
         ):
        result = handler({}, task_id="turn-a")

    assert order == ["lease", "broker"]
    assert result == '{"result": "[]"}'


def test_denial_and_open_breaker_do_zero_broker_work() -> None:
    name = "codex-computer-use"
    mcp_tool._pinned_lazy_server_configs[name] = _config()
    denied = mcp_tool._make_tool_handler(name, "click", 5.0)
    broker = SimpleNamespace(call=lambda *_a, **_kw: pytest.fail("broker called"))
    with patch("tools.mcp_tool._get_codex_cua_broker", return_value=broker), \
         patch("tools.mcp_tool._acquire_single_writer_lease") as lease, \
         patch("tools.mcp_tool._trust_gate_check", return_value='{"error":"denied"}'):
        assert denied({"app": "Finder"}, task_id="turn-a") == '{"error":"denied"}'
    lease.assert_not_called()

    mcp_tool._server_error_counts[name] = mcp_tool._CIRCUIT_BREAKER_THRESHOLD
    mcp_tool._server_breaker_opened_at[name] = __import__("time").monotonic()
    read = mcp_tool._make_tool_handler(name, "list_apps", 5.0)
    with patch("tools.mcp_tool._get_codex_cua_broker", return_value=broker), \
         patch("tools.mcp_tool._trust_gate_check", return_value=None), \
         patch("tools.mcp_tool._acquire_single_writer_lease", return_value=True):
        assert "Auto-retry" in read({}, task_id="turn-a")
    mcp_tool._server_error_counts.pop(name, None)
    mcp_tool._server_breaker_opened_at.pop(name, None)


def test_first_pinned_broker_call_stays_serialized_until_task_release(
    monkeypatch, tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    alias = "first-call-cua"
    observed = []

    class Broker:
        def call(self, tool_name, arguments, *, timeout):
            assert tool_name == "list_apps"
            assert arguments == {}
            observed.append(
                mcp_tool._acquire_single_writer_lease(
                    alias, "second-owner", wait_timeout_seconds=0,
                )
            )
            return {"content": [{"type": "text", "text": "[]"}], "isError": False}

    monkeypatch.setattr(
        mcp_tool, "_machine_account_home_for_lock", lambda: tmp_path
    )
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True), \
         patch("tools.mcp_tool._get_codex_cua_broker", return_value=Broker()), \
         patch("tools.mcp_tool._get_connected_server_for_call") as upstream:
        names = mcp_tool.register_mcp_servers({alias: _config()})
        list_apps = next(name for name in names if name.endswith("__list_apps"))

        result = registry.dispatch(list_apps, {}, task_id="first-owner")
        assert result == '{"result": "[]"}'
        upstream.assert_not_called()
        assert observed == [False]
        assert not mcp_tool._acquire_single_writer_lease(
            alias, "second-owner", wait_timeout_seconds=0,
        )

        mcp_tool.release_mcp_single_writer_leases("first-owner")
        assert alias not in mcp_tool._servers
        assert mcp_tool._acquire_single_writer_lease(
            alias, "second-owner", wait_timeout_seconds=0,
        )
        mcp_tool.release_mcp_single_writer_leases("second-owner")


def test_reconnect_cannot_replace_host_pinned_handlers() -> None:
    server = mcp_tool.MCPServerTask("codex-computer-use")
    mcp_tool._pinned_lazy_server_configs[server.name] = _config()
    mcp_tool._servers[server.name] = server
    server._ready.set()

    with patch("tools.mcp_tool._register_server_tools") as register:
        server._register_discovered_tools_if_needed()

    register.assert_not_called()
    assert server._registered_tool_names == []


def test_release_disconnects_ephemeral_server_before_unlock(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(mcp_tool, "_machine_account_home_for_lock", lambda: tmp_path)
    mcp_tool._configure_single_writer_server(
        "codex-computer-use", enabled=True, wait_timeout_seconds=0,
        capability_key="openai-codex-cua",
    )
    mcp_tool._pinned_lazy_server_configs["codex-computer-use"] = _config()
    assert mcp_tool._acquire_single_writer_lease(
        "codex-computer-use", "owner-a", wait_timeout_seconds=0,
    )
    observed = []

    class Server:
        async def shutdown(self):
            observed.append(
                mcp_tool._acquire_single_writer_lease(
                    "codex-computer-use", "owner-b", wait_timeout_seconds=0,
                )
            )

    mcp_tool._servers["codex-computer-use"] = Server()
    with patch("tools.mcp_tool._run_on_mcp_loop", side_effect=lambda fn, **_kw: __import__('asyncio').run(fn())):
        mcp_tool.release_mcp_single_writer_leases("owner-a")

    assert observed == [False]
    assert mcp_tool._acquire_single_writer_lease(
        "codex-computer-use", "owner-b", wait_timeout_seconds=0,
    )
    assert "codex-computer-use" not in mcp_tool._servers
