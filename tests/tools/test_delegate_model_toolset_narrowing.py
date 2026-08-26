"""Model-facing exact toolset narrowing for delegated UI workers."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from tools import delegate_tool
import run_agent


def _parent(toolsets):
    return SimpleNamespace(
        enabled_toolsets=list(toolsets),
        disabled_toolsets=[],
        valid_tool_names={"delegate_task", "mcp__codex_computer_use__click"},
        model="claude-opus-5",
        provider="anthropic",
        base_url="https://example.invalid",
        api_key="test-only",
        api_mode="anthropic_messages",
        platform="cli",
        session_id="parent-session",
        _session_db=None,
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=threading.Lock(),
        _print_fn=None,
        tool_progress_callback=None,
        thinking_callback=None,
    )


def _creds():
    return {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": None,
        "api_key": None,
        "api_mode": "codex_responses",
        "request_overrides": None,
        "max_output_tokens": None,
        "command": None,
        "args": None,
    }


def test_delegate_schema_exposes_exact_toolset_narrowing():
    prop = delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["toolsets"]

    assert prop["type"] == "array"
    assert prop["items"] == {"type": "string"}
    assert prop["minItems"] == 1
    assert "cannot grant" in prop["description"]


def test_delegate_routes_only_requested_parent_toolset(monkeypatch):
    captured = []
    child = SimpleNamespace(
        _delegate_role="leaf",
        _subagent_id="child-1",
        session_id="child-session",
        tool_progress_callback=None,
    )
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(
        delegate_tool, "_resolve_delegation_credentials", lambda *_a, **_k: _creds()
    )

    def build(**kwargs):
        captured.append(kwargs)
        return child

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        delegate_tool,
        "_run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "_child_role": None,
        },
    )

    result = json.loads(
        delegate_tool.delegate_task(
            goal="Operate the already-running Chrome window",
            toolsets=["mcp-codex-computer-use"],
            parent_agent=_parent(
                ["delegation", "computer_use", "mcp-codex-computer-use"]
            ),
        )
    )

    assert "error" not in result
    assert captured[0]["toolsets"] == ["mcp-codex-computer-use"]
    assert captured[0]["exact_toolsets"] is True
    assert captured[0]["override_provider"] == "openai-codex"
    assert captured[0]["model"] == "gpt-5.6-sol"


def test_delegate_rejects_toolset_not_owned_by_parent(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(
        delegate_tool, "_resolve_delegation_credentials", lambda *_a, **_k: _creds()
    )

    result = json.loads(
        delegate_tool.delegate_task(
            goal="Try an unavailable surface",
            toolsets=["mcp-not-owned"],
            parent_agent=_parent(["delegation", "mcp-codex-computer-use"]),
        )
    )

    assert "error" in result
    assert "not enabled for the parent" in result["error"]


def test_delegate_rejects_empty_or_malformed_toolset_list(monkeypatch):
    parent = _parent(["delegation", "mcp-codex-computer-use"])

    empty = json.loads(
        delegate_tool.delegate_task(
            goal="Do not inherit everything", toolsets=[], parent_agent=parent
        )
    )
    malformed = json.loads(
        delegate_tool.delegate_task(
            goal="Do not accept malformed scope",
            toolsets=["mcp-codex-computer-use", ""],
            parent_agent=parent,
        )
    )

    assert "error" in empty
    assert "non-empty list" in empty["error"]
    assert "error" in malformed
    assert "non-empty strings" in malformed["error"]


def test_live_agent_dispatch_forwards_exact_toolset_scope(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        delegate_tool,
        "delegate_task",
        lambda **kwargs: captured.update(kwargs) or "{}",
    )
    parent = _parent(["delegation", "mcp-codex-computer-use"])

    run_agent.AIAgent._dispatch_delegate_task(
        parent,
        {
            "goal": "Use the signed desktop tool",
            "toolsets": ["mcp-codex-computer-use"],
            "computer_scope": {
                "proposal": {
                    "app": "Google Chrome",
                    "state_digest": "a" * 64,
                    "tool": "click",
                    "args": {"app": "Google Chrome", "element_id": 4},
                }
            },
        },
    )

    assert captured["toolsets"] == ["mcp-codex-computer-use"]
    assert captured["computer_scope"]["proposal"]["state_digest"] == "a" * 64


def test_parent_issues_private_bounded_grant_and_leaf_cannot_escalate(monkeypatch):
    issued = []
    built = []
    child = SimpleNamespace(
        _delegate_role="leaf",
        _subagent_id="child-private",
        session_id="child-session",
        tool_progress_callback=None,
    )
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(
        delegate_tool, "_resolve_delegation_credentials", lambda *_a, **_k: _creds()
    )
    def build(**kwargs):
        built.append(kwargs)
        return child

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        "tools.mcp_tool._issue_mcp_exact_action_grant",
        lambda **kwargs: issued.append(kwargs),
    )
    monkeypatch.setattr(
        delegate_tool,
        "_run_single_child",
        lambda *_a, **_k: {
            "task_index": 0,
            "status": "completed",
            "summary": "done",
            "api_calls": 1,
            "duration_seconds": 0.1,
            "_child_role": None,
        },
    )

    result = json.loads(
        delegate_tool.delegate_task(
            goal="Routine Chrome navigation",
            toolsets=["mcp-codex-computer-use"],
            computer_scope={
                "proposal": {
                    "app": "Google Chrome",
                    "state_digest": "a" * 64,
                    "tool": "click",
                    "args": {"app": "Google Chrome", "element_id": 4},
                }
            },
            parent_agent=_parent(["delegation", "mcp-codex-computer-use"]),
        )
    )
    escalated = json.loads(
        delegate_tool.delegate_task(
            goal="Try to mint high impact",
            toolsets=["mcp-codex-computer-use"],
            computer_scope={"proposal": {
                "app": "Google Chrome",
                "state_digest": "a" * 64,
                "tool": "click",
                "args": {"app": "Google Chrome", "element_id": 4},
                "risk_scope": "high",
            }},
            parent_agent=_parent(["delegation", "mcp-codex-computer-use"]),
        )
    )

    assert "error" not in result
    assert issued == [{
        "task_id": "child-private",
        "server_name": "codex-computer-use",
        "app": "Google Chrome",
        "state_digest": "a" * 64,
        "tool_name": "click",
        "arguments": {"app": "Google Chrome", "element_id": 4},
        "ttl_seconds": 30.0,
    }]
    assert "HOST-BOUND EXACT ACTION" in built[0]["context"]
    assert '"element_id":4' in built[0]["context"]
    assert "state_digest exactly matches" in built[0]["context"]
    assert "error" in escalated
    assert "unsupported authorization fields" in escalated["error"]


def test_computer_scope_rejects_batch_before_admission_build_or_grant(monkeypatch):
    events = []
    parent = _parent(["delegation", "mcp-codex-computer-use"])
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(
        delegate_tool, "_resolve_delegation_credentials", lambda *_a, **_k: _creds()
    )
    monkeypatch.setattr(
        "tools.delegation_admission.validate_spawn",
        lambda **_kwargs: events.append("admitted"),
    )
    monkeypatch.setattr(
        "tools.delegation_live_log.create_live_transcripts",
        lambda *_a, **_k: events.append("live") or (None, [], []),
    )
    monkeypatch.setattr(
        delegate_tool,
        "_build_child_preserving_parent_tools",
        lambda **_kwargs: events.append("built"),
    )
    monkeypatch.setattr(
        "tools.mcp_tool._issue_mcp_exact_action_grant",
        lambda **_kwargs: events.append("granted"),
    )

    result = json.loads(
        delegate_tool.delegate_task(
            tasks=[
                {"goal": "Inspect the currently visible Chrome page state"},
                {"goal": "Inspect the currently visible Chrome tab title"},
            ],
            toolsets=["mcp-codex-computer-use"],
            computer_scope={
                "proposal": {
                    "app": "Google Chrome",
                    "state_digest": "a" * 64,
                    "tool": "click",
                    "args": {"app": "Google Chrome", "element_id": 4},
                }
            },
            parent_agent=parent,
        )
    )

    assert "error" in result
    assert "exactly one execution child" in result["error"]
    assert events == []
    assert parent._active_children == []


def test_public_child_builder_enforces_curated_codex_cua_policy(monkeypatch):
    captured = {}

    class Child:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = "curated-policy-child"

    parent = _parent(["delegation", "mcp-codex-computer-use"])
    monkeypatch.setattr(run_agent, "AIAgent", Child)
    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})

    delegate_tool._build_child_preserving_parent_tools(
        task_index=0,
        goal="Operate Chrome",
        context=None,
        toolsets=["mcp-codex-computer-use"],
        exact_toolsets=True,
        model="gpt-5.6-sol",
        max_iterations=3,
        task_count=1,
        parent_agent=parent,
        override_provider="openai-codex",
        role="leaf",
    )

    policy = captured["ephemeral_system_prompt"]
    for phrase in (
        "fresh state → one indexed action → fresh state",
        "untrusted input",
        "CAPTCHA",
        "already-running normal Chrome profile",
        "parent confirmation",
        "exactly one action",
    ):
        assert phrase in policy
