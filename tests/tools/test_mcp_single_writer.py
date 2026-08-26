"""Crash-safe cross-process single-writer leases for stateful MCP servers."""

from __future__ import annotations

from tools import mcp_tool
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def setup_function():
    mcp_tool._reset_single_writer_leases_for_tests()


def teardown_function():
    mcp_tool._reset_single_writer_leases_for_tests()


def test_single_writer_lease_is_reentrant_for_one_task_and_blocks_another():
    mcp_tool._configure_single_writer_server(
        "desktop", enabled=True, idle_timeout_seconds=60
    )

    assert mcp_tool._acquire_single_writer_lease(
        "desktop", "task-a", wait_timeout_seconds=0
    )
    assert mcp_tool._acquire_single_writer_lease(
        "desktop", "task-a", wait_timeout_seconds=0
    )
    assert not mcp_tool._acquire_single_writer_lease(
        "desktop", "task-b", wait_timeout_seconds=0
    )


def test_release_allows_the_next_task_to_take_the_server():
    mcp_tool._configure_single_writer_server(
        "desktop", enabled=True, idle_timeout_seconds=60
    )
    assert mcp_tool._acquire_single_writer_lease(
        "desktop", "task-a", wait_timeout_seconds=0
    )

    mcp_tool.release_mcp_single_writer_leases("task-a")

    assert mcp_tool._acquire_single_writer_lease(
        "desktop", "task-b", wait_timeout_seconds=0
    )


def test_active_owner_is_never_stolen_based_on_idle_time():
    now = [100.0]
    mcp_tool._configure_single_writer_server(
        "desktop", enabled=True, idle_timeout_seconds=5
    )
    assert mcp_tool._acquire_single_writer_lease(
        "desktop", "task-a", wait_timeout_seconds=0, monotonic=lambda: now[0]
    )

    now[0] = 106.0

    assert not mcp_tool._acquire_single_writer_lease(
        "desktop", "task-b", wait_timeout_seconds=0, monotonic=lambda: now[0]
    )


def test_unconfigured_servers_do_not_create_a_lease():
    assert mcp_tool._acquire_single_writer_lease(
        "ordinary", "task-a", wait_timeout_seconds=0
    )
    assert mcp_tool._acquire_single_writer_lease(
        "ordinary", "task-b", wait_timeout_seconds=0
    )
    assert mcp_tool._single_writer_leases == {}


def test_registration_applies_bounded_single_writer_mapping_without_connecting():
    mcp_tool.register_mcp_servers(
        {
            "desktop": {
                "enabled": False,
                "single_writer": {
                    "enabled": True,
                    "idle_timeout_seconds": 240,
                    "wait_timeout_seconds": 4,
                },
            }
        }
    )

    assert mcp_tool._single_writer_policies["desktop"] == (240.0, 4.0)


def test_invalid_single_writer_timeouts_fail_closed_to_disabled():
    mcp_tool.register_mcp_servers(
        {
            "desktop": {
                "enabled": False,
                "single_writer": {
                    "enabled": True,
                    "idle_timeout_seconds": "not-a-number",
                },
            }
        }
    )

    assert "desktop" not in mcp_tool._single_writer_policies


def test_cross_process_contention_and_crash_recovery(tmp_path, monkeypatch):
    server_name = f"desktop-{uuid.uuid4().hex}"
    account_home = tmp_path / "os-account-home"
    account_home.mkdir()
    monkeypatch.setattr(
        mcp_tool, "_machine_account_home_for_lock", lambda: account_home
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "parent-profile"))
    script = f"""
from tools import mcp_tool
mcp_tool._machine_account_home_for_lock = lambda: __import__('pathlib').Path({str(account_home)!r})
mcp_tool._configure_single_writer_server({server_name!r}, enabled=True, wait_timeout_seconds=0)
assert mcp_tool._acquire_single_writer_lease({server_name!r}, 'child', wait_timeout_seconds=0)
print('ready', flush=True)
input()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["HERMES_HOME"] = str(tmp_path / "child-profile")
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        mcp_tool._configure_single_writer_server(
            server_name, enabled=True, wait_timeout_seconds=0
        )
        assert not mcp_tool._acquire_single_writer_lease(
            server_name, "parent", wait_timeout_seconds=0
        )
        child.kill()
        child.wait(timeout=5)
        assert mcp_tool._acquire_single_writer_lease(
            server_name, "parent", wait_timeout_seconds=1
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_malformed_reload_does_not_delete_an_active_lease():
    mcp_tool._configure_single_writer_server(
        "desktop", enabled=True, wait_timeout_seconds=0
    )
    assert mcp_tool._acquire_single_writer_lease(
        "desktop", "task-a", wait_timeout_seconds=0
    )

    mcp_tool.register_mcp_servers(
        {
            "desktop": {
                "enabled": False,
                "single_writer": {
                    "enabled": True,
                    "wait_timeout_seconds": "invalid",
                },
            }
        }
    )

    assert not mcp_tool._acquire_single_writer_lease(
        "desktop", "task-b", wait_timeout_seconds=0
    )


def test_malformed_scalar_single_writer_rejects_server_fail_closed():
    mcp_tool.register_mcp_servers({
        "desktop": {"enabled": False, "single_writer": "garbage"}
    })

    assert "desktop" not in mcp_tool._single_writer_policies
    assert "desktop" not in mcp_tool._lazy_server_configs
    assert "desktop" not in mcp_tool._servers
    assert "invalid single_writer" in mcp_tool._server_connect_errors["desktop"]


def test_writer_lock_missing_capability_key_fails_closed_without_identity_lookup(
    monkeypatch,
):
    mcp_tool._single_writer_policies["missing-key"] = (90.0, 0.0)
    monkeypatch.setattr(
        mcp_tool,
        "_mcp_capability_identity",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("writer condition must not re-enter MCP state")
        ),
    )

    assert not mcp_tool._acquire_single_writer_lease(
        "missing-key", "owner", wait_timeout_seconds=0
    )
    assert "missing-key" not in mcp_tool._single_writer_leases
    assert "missing-key" not in mcp_tool._single_writer_process_locks


def test_codex_cua_adapter_identity_makes_server_aliases_share_lock(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        mcp_tool, "_machine_account_home_for_lock", lambda: tmp_path
    )
    compatibility = {
        "app_server_catalog_sha256": "f710c1eacba2487b5547ddafe8aeb616268850ea4501df3a4a047552a1608a40",
        "tools_sha256": "dd485a140f5fbebe14147fb3ee2ed3914618b3484964efe02262b2479b322f1d",
        "capabilities_sha256": "52aa21370a62916d63adb5718fa1be519ec0fe4390136bf36e701be54e5582a5",
        "tool_count": 10,
        "tools_only": True,
    }
    tools = {
        "include": [
            "list_apps", "get_app_state", "click", "perform_secondary_action",
            "set_value", "select_text", "scroll", "drag", "press_key", "type_text",
        ],
        "resources": False,
        "prompts": False,
    }
    cua_config = {
        "enabled": False,
        "transport": "codex_app_server",
        "single_writer": True,
        "supports_parallel_tool_calls": False,
        "minimal_env": True,
        "trust": "untrusted",
        "compatibility": compatibility,
        "tools": tools,
    }
    mcp_tool.register_mcp_servers({
        "codex-computer-use": dict(cua_config),
        "renamed-local-cua": dict(cua_config),
    })

    assert mcp_tool._single_writer_capability_keys["codex-computer-use"] == (
        mcp_tool._single_writer_capability_keys["renamed-local-cua"]
    )
    assert mcp_tool._acquire_single_writer_lease(
        "codex-computer-use", "task-a", wait_timeout_seconds=0
    )
    assert not mcp_tool._acquire_single_writer_lease(
        "renamed-local-cua", "task-b", wait_timeout_seconds=0
    )


def test_handler_blocks_a_second_owner_before_any_transport(monkeypatch):
    mcp_tool._configure_single_writer_server(
        "desktop", enabled=True, idle_timeout_seconds=60, wait_timeout_seconds=0
    )
    assert mcp_tool._acquire_single_writer_lease(
        "desktop", "task-a", wait_timeout_seconds=0
    )
    monkeypatch.setattr(
        mcp_tool,
        "_trust_gate_check",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mcp_tool,
        "_get_connected_server_for_call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a blocked owner must not reach transport")
        ),
    )
    handler = mcp_tool._make_tool_handler("desktop", "click", 5)

    result = json.loads(handler({}, task_id="task-b"))

    assert "error" in result
    assert "reserved by another agent task" in result["error"]
