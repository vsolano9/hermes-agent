"""Exact-action authorization for delegated Codex Computer Use calls."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

from tools import mcp_tool


STATE_DIGEST = "a" * 64
NEXT_STATE_DIGEST = "b" * 64
ACTION_ARGS = {"app": "Google Chrome", "element_id": 42, "button": "left"}


def setup_function():
    mcp_tool._reset_mcp_routine_grants_for_tests()
    mcp_tool._server_trust_levels["codex"] = "untrusted"
    mcp_tool._tool_read_only_hints["codex"] = {
        "get_app_state": True,
        "click": False,
        "type_text": False,
    }


def teardown_function():
    mcp_tool._reset_mcp_routine_grants_for_tests()


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
