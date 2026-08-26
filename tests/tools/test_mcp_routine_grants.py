"""Exact-action authorization for delegated Codex Computer Use calls."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import mcp_tool
from tools.registry import ToolRegistry


STATE_DIGEST = "a" * 64
NEXT_STATE_DIGEST = "b" * 64
ACTION_ARGS = {"app": "Google Chrome", "element_id": 42, "button": "left"}
_TEST_CUA_ALIASES = {
    "parent-cua-alias",
    "child-cua-alias",
    "state-cua-alias",
}


def _clear_test_alias_state() -> None:
    for server_name in _TEST_CUA_ALIASES:
        mcp_tool._pinned_lazy_server_configs.pop(server_name, None)
        mcp_tool._pinned_lazy_server_tool_names.pop(server_name, None)
        mcp_tool._mcp_server_capability_identities.pop(server_name, None)
        mcp_tool._server_trust_levels.pop(server_name, None)
        mcp_tool._tool_read_only_hints.pop(server_name, None)
        mcp_tool._parallel_safe_servers.discard(server_name)
        mcp_tool._server_connect_errors.pop(server_name, None)
    for tool_name, server_name in list(mcp_tool._mcp_tool_server_names.items()):
        if server_name in _TEST_CUA_ALIASES:
            mcp_tool._mcp_tool_server_names.pop(tool_name, None)
    mcp_tool._reset_single_writer_leases_for_tests()


def _cua_config():
    tools = [
        "list_apps", "get_app_state", "click", "perform_secondary_action",
        "set_value", "select_text", "scroll", "drag", "press_key", "type_text",
    ]
    return {
        "transport": "codex_app_server",
        "trust": "untrusted",
        "supports_parallel_tool_calls": False,
        "single_writer": True,
        "minimal_env": True,
        "compatibility": {
            "app_server_catalog_sha256": "bc4f6aca3e12fecaa3d7eee6c800f7885790c3cdebe27b84e2e1ca5d3a020c38",
            "tools_sha256": "dd485a140f5fbebe14147fb3ee2ed3914618b3484964efe02262b2479b322f1d",
            "capabilities_sha256": "52aa21370a62916d63adb5718fa1be519ec0fe4390136bf36e701be54e5582a5",
            "tool_count": 10,
            "tools_only": True,
        },
        "tools": {"include": tools, "resources": False, "prompts": False},
    }


def setup_function():
    _clear_test_alias_state()
    mcp_tool._reset_mcp_routine_grants_for_tests()
    mcp_tool._remember_mcp_capability_identity(
        "codex-computer-use", _cua_config()
    )
    mcp_tool._remember_mcp_capability_identity(
        "openai-codex-cua", _cua_config()
    )
    mcp_tool._server_trust_levels["codex"] = "untrusted"
    mcp_tool._tool_read_only_hints["codex"] = {
        "get_app_state": True,
        "click": False,
        "type_text": False,
    }


def teardown_function():
    mcp_tool._reset_mcp_routine_grants_for_tests()
    _clear_test_alias_state()


def _issue(*, task_id="child-1", app="Google Chrome", tool="click", args=None,
           state_digest=STATE_DIGEST, ttl_seconds=30, monotonic=None):
    kwargs = {
        "task_id": task_id,
        "server_name": "codex",
        "app": app,
        "state_digest": state_digest,
        "tool_name": tool,
        "arguments": ACTION_ARGS if args is None else args,
        "ttl_seconds": ttl_seconds,
    }
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    mcp_tool._issue_mcp_exact_action_grant(**kwargs)


def _observe(*, task_id="child-1", app="Google Chrome", digest=STATE_DIGEST):
    mcp_tool._record_mcp_state_observation(
        task_id=task_id,
        server_name="codex",
        app=app,
        state_digest=digest,
    )


def _deny_and_count(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tools.approval.request_elicitation_consent",
        lambda *_a, **_k: calls.append("asked") or "deny",
    )
    return calls


def test_exact_proposed_action_avoids_popup_once(monkeypatch):
    _observe()
    _issue()
    monkeypatch.setattr(
        "tools.approval.request_elicitation_consent",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("an exact parent grant must not show approval")
        ),
    )

    assert mcp_tool._trust_gate_check("codex", "click", ACTION_ARGS, "child-1") is None


def test_argument_or_tool_change_cannot_use_grant(monkeypatch):
    calls = _deny_and_count(monkeypatch)
    _observe()
    _issue()

    assert mcp_tool._trust_gate_check(
        "codex", "click", {**ACTION_ARGS, "element_id": 43}, "child-1"
    )
    assert mcp_tool._trust_gate_check(
        "codex", "type_text", ACTION_ARGS, "child-1"
    )
    assert calls == ["asked", "asked"]


def test_state_digest_mismatch_cannot_use_grant(monkeypatch):
    calls = _deny_and_count(monkeypatch)
    _observe(digest=NEXT_STATE_DIGEST)
    _issue(state_digest=STATE_DIGEST)

    assert mcp_tool._trust_gate_check("codex", "click", ACTION_ARGS, "child-1")
    assert calls == ["asked"]


def test_grant_is_single_use_and_requires_fresh_state_before_next_action(monkeypatch):
    calls = _deny_and_count(monkeypatch)
    _observe()
    _issue()
    assert mcp_tool._trust_gate_check("codex", "click", ACTION_ARGS, "child-1") is None

    assert mcp_tool._trust_gate_check("codex", "click", ACTION_ARGS, "child-1")
    _issue()
    assert mcp_tool._trust_gate_check("codex", "click", ACTION_ARGS, "child-1")

    _observe(digest=NEXT_STATE_DIGEST)
    _issue(state_digest=NEXT_STATE_DIGEST)
    assert mcp_tool._trust_gate_check("codex", "click", ACTION_ARGS, "child-1") is None
    assert calls == ["asked", "asked"]


def test_expired_wrong_app_and_wrong_task_cannot_bypass(monkeypatch):
    calls = _deny_and_count(monkeypatch)
    _observe()
    _issue(monotonic=lambda: 100.0)

    assert mcp_tool._trust_gate_check(
        "codex", "click", {**ACTION_ARGS, "app": "Safari"}, "child-1",
        monotonic=lambda: 101.0,
    )
    assert mcp_tool._trust_gate_check(
        "codex", "click", ACTION_ARGS, "child-2", monotonic=lambda: 101.0
    )
    assert mcp_tool._trust_gate_check(
        "codex", "click", ACTION_ARGS, "child-1", monotonic=lambda: 131.0
    )
    assert calls == ["asked", "asked", "asked"]


def test_no_grant_retains_untrusted_approval_before_transport(monkeypatch):
    calls = _deny_and_count(monkeypatch)

    result = mcp_tool._trust_gate_check(
        "codex", "click", ACTION_ARGS, "child-raw"
    )

    assert "error" in json.loads(result)
    assert calls == ["asked"]


def test_successful_state_result_exposes_and_records_stable_digest():
    raw = json.dumps({"result": {"windows": [{"title": "Work"}]}})

    decorated = mcp_tool._record_state_result_for_exact_grant(
        server_name="codex-computer-use",
        tool_name="get_app_state",
        args={"app": "Google Chrome"},
        task_id="child-1",
        result=raw,
    )
    payload = json.loads(decorated)
    digest = payload["state_digest"]
    assert len(digest) == 64

    mcp_tool._issue_mcp_exact_action_grant(
        task_id="child-1",
        server_name="codex-computer-use",
        app="Google Chrome",
        state_digest=digest,
        tool_name="click",
        arguments=ACTION_ARGS,
        ttl_seconds=30,
    )
    mcp_tool._server_trust_levels["codex-computer-use"] = "untrusted"
    mcp_tool._tool_read_only_hints["codex-computer-use"] = {"click": False}
    assert mcp_tool._trust_gate_check(
        "codex-computer-use", "click", ACTION_ARGS, "child-1"
    ) is None


def test_canonical_adapter_aliases_share_state_and_exact_action_grant(monkeypatch):
    registry = ToolRegistry()
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        mcp_tool.register_mcp_servers({"parent-cua-alias": _cua_config()})
        mcp_tool.register_mcp_servers({"child-cua-alias": _cua_config()})

    mcp_tool._record_mcp_state_observation(
        task_id="child-alias",
        server_name="child-cua-alias",
        app="Google Chrome",
        state_digest=STATE_DIGEST,
    )
    mcp_tool._issue_mcp_exact_action_grant(
        task_id="child-alias",
        server_name="parent-cua-alias",
        app="Google Chrome",
        state_digest=STATE_DIGEST,
        tool_name="click",
        arguments=ACTION_ARGS,
        ttl_seconds=30,
    )
    monkeypatch.setattr(
        "tools.approval.request_elicitation_consent",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("canonical alias grant must avoid popup")
        ),
    )

    assert mcp_tool._trust_gate_check(
        "child-cua-alias", "click", ACTION_ARGS, "child-alias"
    ) is None


def test_state_digest_is_decorated_for_canonical_adapter_alias() -> None:
    registry = ToolRegistry()
    with patch("tools.registry.registry", registry), \
         patch("tools.mcp_tool._MCP_AVAILABLE", True):
        mcp_tool.register_mcp_servers({"state-cua-alias": _cua_config()})

    decorated = mcp_tool._record_state_result_for_exact_grant(
        server_name="state-cua-alias",
        tool_name="get_app_state",
        args={"app": "Google Chrome"},
        task_id="child-state",
        result=json.dumps({"result": {"window": "Work"}}),
    )

    assert len(json.loads(decorated)["state_digest"]) == 64


def test_state_digest_never_opens_media_paths_from_untrusted_result_text(
    tmp_path, monkeypatch
):
    regular = tmp_path / "oversized.bin"
    with regular.open("wb") as handle:
        handle.truncate(128 * 1024 * 1024)
    fifo = tmp_path / "blocking.fifo"
    os.mkfifo(fifo)
    symlink = tmp_path / "passwd-link"
    symlink.symlink_to("/etc/passwd")
    media_text = "\n".join(
        f"MEDIA:{path}"
        for path in ("/etc/passwd", "/dev/zero", fifo, symlink, regular)
    )
    raw = json.dumps({"result": media_text})
    opened = []

    def reject_read(path):
        opened.append(str(path))
        raise AssertionError("untrusted MEDIA text must never trigger filesystem I/O")

    monkeypatch.setattr(Path, "read_bytes", reject_read)
    monkeypatch.setattr(io, "open", lambda *a, **_k: reject_read(a[0]))
    monkeypatch.setattr(os, "open", lambda *a, **_k: reject_read(a[0]))

    decorated = mcp_tool._record_state_result_for_exact_grant(
        server_name="codex-computer-use",
        tool_name="get_app_state",
        args={"app": "Google Chrome"},
        task_id="child-1",
        result=raw,
    )

    expected = hashlib.sha256(
        json.dumps(
            {"result": media_text},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert opened == []
    assert json.loads(decorated)["state_digest"] == expected


def test_trusted_raw_result_digest_is_independent_of_rendered_cache_path():
    raw_result = SimpleNamespace(
        content=[SimpleNamespace(type="image", data="c2FtZS1zY3JlZW5zaG90")],
        structuredContent={"windows": [{"title": "Work"}]},
        isError=False,
    )
    trusted_digest = mcp_tool._trusted_mcp_result_sha256(raw_result)

    one = mcp_tool._record_state_result_for_exact_grant(
        server_name="codex-computer-use",
        tool_name="get_app_state",
        args={"app": "Google Chrome"},
        task_id="child-1",
        result=json.dumps({"result": "state\nMEDIA:/cache/random-one.png"}),
        trusted_state_digest=trusted_digest,
    )
    two = mcp_tool._record_state_result_for_exact_grant(
        server_name="codex-computer-use",
        tool_name="get_app_state",
        args={"app": "Google Chrome"},
        task_id="child-2",
        result=json.dumps({"result": "state\nMEDIA:/cache/random-two.png"}),
        trusted_state_digest=trusted_digest,
    )

    assert json.loads(one)["state_digest"] == json.loads(two)["state_digest"]
