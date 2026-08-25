"""Tests for delegate_tool toolset scoping.

Verifies that subagents cannot gain tools that the parent does not have.
The LLM controls the `toolsets` parameter — without intersection with the
parent's enabled_toolsets, it can escalate privileges by requesting
arbitrary toolsets.
"""

import os
import logging
import threading
from types import SimpleNamespace

import model_tools
import run_agent
from tools import delegate_tool
from tools.delegate_tool import _strip_blocked_tools, _emit_parent_console


def test_host_exact_toolsets_resolve_model_visible_tools_without_legacy_drift(
    monkeypatch,
):
    """Exact mode is host-only; legacy empty still inherits parent tools."""
    built = []

    class ResolvingAgent:
        def __init__(self, **kwargs):
            self.enabled_toolsets = list(kwargs["enabled_toolsets"])
            self.disabled_toolsets = list(kwargs["disabled_toolsets"])
            self.tools = model_tools.get_tool_definitions(
                enabled_toolsets=self.enabled_toolsets,
                disabled_toolsets=self.disabled_toolsets,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            self.valid_tool_names = {
                item["function"]["name"] for item in self.tools
            }
            self.session_id = f"child-{len(built)}"
            built.append(self)

    class Parent:
        enabled_toolsets = ["file", "terminal", "delegation", "hermes-cli"]
        disabled_toolsets = []
        valid_tool_names = {"read_file", "terminal", "delegate_task"}
        model = "synthetic-model"
        provider = "synthetic-provider"
        base_url = "https://example.invalid/v1"
        api_key = "synthetic-key"
        api_mode = "chat_completions"
        platform = "cli"
        session_id = "exact-parent"
        _session_db = None
        _delegate_depth = 0
        _active_children = []
        _active_children_lock = threading.Lock()
        _print_fn = None
        tool_progress_callback = None
        thinking_callback = None

    parent = Parent()
    original_parent_toolsets = list(parent.enabled_toolsets)
    monkeypatch.setattr(run_agent, "AIAgent", ResolvingAgent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(delegate_tool, "_get_orchestrator_enabled", lambda: True)
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 3)

    def build(toolsets, *, exact=False, role="leaf"):
        return delegate_tool._build_child_preserving_parent_tools(
            task_index=len(built),
            goal="inspect safely",
            context=None,
            toolsets=toolsets,
            exact_toolsets=exact,
            model=None,
            max_iterations=3,
            task_count=1,
            parent_agent=parent,
            role=role,
        )

    inherited_none = build(None)
    inherited_empty = build([])
    exact_empty = build([], exact=True, role="orchestrator")
    exact_file = build(["file"], exact=True, role="orchestrator")
    exact_composite_leaf = build(["hermes-cli"], exact=True, role="leaf")
    exact_composite_orchestrator = build(
        ["hermes-cli"], exact=True, role="orchestrator"
    )
    exact_delegation = build(["delegation"], exact=True, role="orchestrator")

    assert inherited_none.valid_tool_names == inherited_empty.valid_tool_names
    assert inherited_none.valid_tool_names
    assert exact_empty.valid_tool_names == set()
    assert exact_empty.tools == []
    assert exact_file.valid_tool_names
    assert exact_file.valid_tool_names == {
        item["function"]["name"]
        for item in model_tools.get_tool_definitions(
            enabled_toolsets=["file"],
            disabled_toolsets=exact_file.disabled_toolsets,
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
    }
    assert "delegate_task" not in exact_file.valid_tool_names
    assert "delegate_task" not in exact_composite_leaf.valid_tool_names
    assert "delegate_task" not in exact_composite_orchestrator.valid_tool_names
    assert "delegate_task" not in exact_delegation.valid_tool_names
    assert parent.enabled_toolsets == original_parent_toolsets


def test_authoritative_routed_exact_empty_never_borrows_parent_runtime_or_tools(
    monkeypatch,
):
    """The instantiated child, not only builder kwargs, is zero-tool/scoped."""
    constructed = []

    class ResolvingAgent:
        def __init__(self, **kwargs):
            self.provider = kwargs["provider"]
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]
            self.api_key = kwargs["api_key"]
            self.api_mode = kwargs["api_mode"]
            self.acp_command = kwargs["acp_command"]
            self.acp_args = list(kwargs["acp_args"])
            self.reasoning_config = kwargs["reasoning_config"]
            self.enabled_toolsets = list(kwargs["enabled_toolsets"])
            self.disabled_toolsets = list(kwargs["disabled_toolsets"])
            self.tools = model_tools.get_tool_definitions(
                enabled_toolsets=self.enabled_toolsets,
                disabled_toolsets=self.disabled_toolsets,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            self.valid_tool_names = {
                item["function"]["name"] for item in self.tools
            }
            self.session_id = "authoritative-child"
            constructed.append(self)

    parent = SimpleNamespace(
        enabled_toolsets=["hermes-cli", "delegation"],
        disabled_toolsets=[],
        valid_tool_names={"terminal", "read_file", "write_file", "delegate_task"},
        model="parent-model",
        provider="parent-provider",
        base_url="https://parent-secret.invalid/v1",
        api_key="parent-secret-key",
        api_mode="anthropic_messages",
        acp_command="parent-acp",
        acp_args=["writeTextFile"],
        reasoning_config={"enabled": True, "effort": "low"},
        platform="cli",
        session_id="authoritative-parent",
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
    )
    parent_toolsets = list(parent.enabled_toolsets)
    parent_cwd = os.getcwd()
    monkeypatch.setattr(run_agent, "AIAgent", ResolvingAgent)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})

    child = delegate_tool._build_child_preserving_parent_tools(
        task_index=0,
        goal="read only",
        context=None,
        toolsets=[],
        exact_toolsets=True,
        model="route-model",
        max_iterations=3,
        task_count=1,
        parent_agent=parent,
        override_provider="route-provider",
        override_base_url="",
        override_api_key="",
        override_api_mode="chat_completions",
        override_request_overrides={},
        override_acp_command=None,
        override_acp_args=[],
        authoritative_route_overrides=True,
        override_reasoning_config={"enabled": False},
        role="orchestrator",
    )

    assert child is constructed[0]
    assert child.provider == "route-provider"
    assert child.model == "route-model"
    assert child.base_url == ""
    assert child.api_key == ""
    assert child.api_mode == "chat_completions"
    assert child.acp_command is None
    assert child.acp_args == []
    assert child.reasoning_config == {"enabled": False}
    assert child.enabled_toolsets == []
    assert child.tools == []
    assert child.valid_tool_names == set()
    assert "delegate_task" not in child.valid_tool_names
    assert parent.enabled_toolsets == parent_toolsets
    assert parent.base_url == "https://parent-secret.invalid/v1"
    assert parent.api_key == "parent-secret-key"
    assert parent.acp_command == "parent-acp"
    assert os.getcwd() == parent_cwd


def test_authoritative_builder_sanitizes_custom_and_generic_pool_debug_logs(
    monkeypatch, caplog
):
    canaries = (
        "https://pool-secret.invalid/private",
        "sk-pool-exception-canary",
        "POOL_ENV_CANARY=value",
        "/private/pool/config.yaml",
        "generic-provider-canary",
    )
    raw_error = " ".join(canaries)

    class Child:
        def __init__(self, **kwargs):
            self.session_id = f"pool-child-{kwargs['provider']}"

    parent = SimpleNamespace(
        enabled_toolsets=["file"],
        disabled_toolsets=[],
        valid_tool_names={"read_file"},
        model="parent-model",
        provider="parent-provider",
        base_url="https://parent.invalid/v1",
        api_key="parent-key",
        api_mode="chat_completions",
        platform="cli",
        session_id="pool-parent",
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
    )
    monkeypatch.setattr(run_agent, "AIAgent", Child)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(
        "agent.credential_pool.get_custom_provider_pool_key",
        lambda *_args: (_ for _ in ()).throw(RuntimeError(raw_error)),
    )
    caplog.set_level(logging.DEBUG, logger="tools.delegate_tool")

    delegate_tool._build_child_preserving_parent_tools(
        task_index=0,
        goal="custom pool probe",
        context=None,
        toolsets=[],
        exact_toolsets=True,
        model="route-model",
        max_iterations=3,
        task_count=1,
        parent_agent=parent,
        override_provider="custom",
        override_base_url=canaries[0],
        override_api_key="safe-test-key",
        override_api_mode="chat_completions",
        authoritative_route_overrides=True,
    )

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda *_args: (_ for _ in ()).throw(RuntimeError(raw_error)),
    )
    delegate_tool._build_child_preserving_parent_tools(
        task_index=1,
        goal="generic pool probe",
        context=None,
        toolsets=[],
        exact_toolsets=True,
        model="route-model",
        max_iterations=3,
        task_count=1,
        parent_agent=parent,
        override_provider=canaries[4],
        override_base_url="https://safe.invalid/v1",
        override_api_key="safe-test-key",
        override_api_mode="chat_completions",
        authoritative_route_overrides=True,
    )

    assert "Custom child credential pool resolution was unavailable." in caplog.text
    assert "Child credential pool resolution was unavailable." in caplog.text
    assert all(canary not in caplog.text for canary in canaries)


class TestToolsetIntersection:
    """Subagent toolsets must be a subset of parent's enabled_toolsets."""

    def test_requested_toolsets_intersected_with_parent(self):
        """LLM requests toolsets parent doesn't have — extras are dropped."""
        parent = SimpleNamespace(enabled_toolsets=["terminal", "file"])

        # Simulate the intersection logic from _build_child_agent
        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["terminal", "file", "web", "browser", "rl"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert sorted(scoped) == ["file", "terminal"]
        assert "web" not in scoped
        assert "browser" not in scoped
        assert "rl" not in scoped


    def test_strip_blocked_removes_delegation(self):
        """Blocked toolsets (delegation, clarify, etc.) are always removed."""
        child = _strip_blocked_tools(["terminal", "delegation", "clarify", "memory"])
        assert "delegation" not in child
        assert "clarify" not in child
        assert "memory" not in child
        assert "terminal" in child

    def test_empty_intersection_yields_empty_toolsets(self):
        """If parent has no overlap with requested, child gets nothing extra."""
        parent = SimpleNamespace(enabled_toolsets=["terminal"])

        parent_toolsets = set(parent.enabled_toolsets)
        requested = ["web", "browser"]
        scoped = [t for t in requested if t in parent_toolsets]

        assert scoped == []


class TestEmitParentConsole:
    """Progress lines (e.g. ``✓ [N/M] …``) must route through the parent's
    configured ``_safe_print`` in headless stdio hosts (ACP, gateway) so
    they don't land on stdout and corrupt JSON-RPC frames. Regression for a
    bug where delegate_task completion lines pushed to stdout caused
    ``Failed to parse JSON message: ✓ [3/3] …`` errors in the ACP adapter."""

    def test_routes_through_parent_safe_print_when_available(self, capsys):
        captured_lines = []
        parent = SimpleNamespace(_safe_print=lambda line: captured_lines.append(line))

        _emit_parent_console(parent, "  ✓ [1/3] Research done  (11.55s)")

        assert captured_lines == ["  ✓ [1/3] Research done  (11.55s)"]
        stdout_stderr = capsys.readouterr()
        assert stdout_stderr.out == ""
        assert stdout_stderr.err == ""


    def test_non_callable_safe_print_is_ignored(self, capsys):
        """Defensive: if _safe_print is set but not callable, fall back."""
        parent = SimpleNamespace(_safe_print="not-a-function")
        _emit_parent_console(parent, "  ✓ [3/3] non-callable guard")
        captured = capsys.readouterr()
        assert "non-callable guard" in captured.out
