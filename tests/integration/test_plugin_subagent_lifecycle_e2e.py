"""External-process E2E for discovered plugin subagent lifecycle contracts.

The pytest process is only a supervisor.  The host mode starts with isolated
environment paths already set, then enters the real gateway/session/agent/tool
pipeline against a loopback OpenAI-compatible provider.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml


_HOST_TIMEOUT = 45


def test_public_task7_contract_is_generic_and_additive():
    from agent.subagent_lifecycle import (
        SubagentRouteAssessment,
        SubagentRouteCatalog,
        SubagentRouteIdentity,
    )
    from hermes_cli.plugin_invocation import PluginToolInvocation

    assert tuple(SubagentRouteIdentity.__dataclass_fields__) == ("provider", "model")
    assert {
        "route", "eligible", "reason", "transport", "authenticated",
        "agent_capable", "exact_empty_model_tools",
        "mutation_evidence_complete", "independent_mutation_channels",
        "hermes_model_tool_count", "assessed_at", "assessment_id",
    } <= set(SubagentRouteAssessment.__dataclass_fields__)
    assert {
        "complete", "routes", "candidate_count", "reason", "assessed_at",
        "snapshot_id",
    } <= set(
        SubagentRouteCatalog.__dataclass_fields__
    )
    assert {
        "execution_kind", "delegation_depth", "delegation_role", "platform",
    } <= set(PluginToolInvocation.__dataclass_fields__)


def _isolated_config(base_url: str) -> dict:
    return {
        "model": {"default": "parent-model", "provider": "custom:parent",
                  "context_length": 131072},
        "custom_providers": [{"name": "parent", "base_url": base_url,
                              "api_key": "synthetic-e2e-only", "model": "parent-model",
                              "api_mode": "chat_completions", "discover_models": False}],
        "plugins": {"enabled": ["lifecycle-e2e", "foreign-e2e"]},
        "platform_toolsets": {"cli": ["plugin_lifecycle_e2e"]},
        "tools": {"tool_search": {"enabled": "off"}},
        "delegation": {"max_spawn_depth": 2},
        "agent": {"max_iterations": 8},
    }


def _plugin_source() -> str:
    return '''
import json, os, time
from pathlib import Path
from agent.subagent_lifecycle import SubagentHandle, SubagentLaunchRequest, SubagentLaunchRequestV2
from hermes_constants import get_hermes_home

ctx_ref = None
handles = []
admission_handles = []
receipt_path = get_hermes_home() / "e2e-plugin-receipts.jsonl"
handle_path = get_hermes_home() / "e2e-handles.json"
nested_handle_path = get_hermes_home() / "e2e-nested-handle.json"

def _record(value):
    with receipt_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\\n")

def _request(label):
    return SubagentLaunchRequestV2(
        api_contract_version=2,
        base=SubagentLaunchRequest(goal="delayed read-only " + label,
                                   model="worker-model",
                                   correlation_id="e2e-" + label),
        toolset_mode="exact", exact_toolsets=(),
        provider="lmstudio", reasoning_effort="high")

def launch(args, *, invocation):
    first = ctx_ref.subagent_lifecycle.launch(_request("ctx"))
    second = invocation.subagents.launch(_request("invocation"))
    handles[:] = [first, second]
    handle_path.write_text(json.dumps([item.to_dict() for item in handles]), encoding="utf-8")
    listed = {item.handle: item for item in invocation.subagents.list()}
    launch_statuses = [listed[first], listed[second]]
    value = {"phase": "launch", "task_id": invocation.task_id,
             "operation_id": invocation.operation_id,
             "session_id": invocation.session_id,
             "states": [item.state.value for item in launch_statuses],
             "audit": [{"launch_operation_id": item.audit_metadata.launch_operation_id,
                         "operation_id": item.audit_metadata.operation_id}
                        if item.audit_metadata else None for item in launch_statuses],
             "handles": [item.to_dict() for item in handles]}
    _record(value)
    return json.dumps(value, sort_keys=True)

def control(args, *, invocation):
    if not handles and handle_path.exists():
        handles.extend(SubagentHandle.from_dict(item)
                       for item in json.loads(handle_path.read_text(encoding="utf-8")))
    base = ctx_ref.subagent_lifecycle
    scoped = invocation.subagents
    before = {"ctx": [item.state.value for item in base.list()],
              "invocation": [item.state.value for item in scoped.list()]}
    status_before = [base.status(handles[0]).state.value,
                     scoped.status(handles[1]).state.value]
    steer = scoped.steer(handles[1], "finish without mutation")
    stop = base.stop(handles[0], reason="e2e cooperative stop")
    deadline = time.monotonic() + 10
    completions = []
    while time.monotonic() < deadline:
        completions = [base.collect(handles[0]), scoped.collect(handles[1])]
        if all(item.ready for item in completions) or all(
                item.diagnostic == "UNKNOWN_HANDLE" for item in completions):
            break
        time.sleep(0.02)
    repeated = [base.collect(handles[0]), scoped.collect(handles[1])]
    value = {"phase": "control", "task_id": invocation.task_id,
             "operation_id": invocation.operation_id,
             "session_id": invocation.session_id, "before": before,
             "status_before": status_before,
             "steer": steer.disposition.value, "stop": stop.accepted,
             "ready": [item.ready for item in completions],
             "events": [item.event_id for item in completions],
             "diagnostics": [item.diagnostic for item in completions],
             "stable": [
                 first.event_id == second.event_id
                 and first.collected_at == second.collected_at
                 and (first.result.result_hash if first.result else None)
                     == (second.result.result_hash if second.result else None)
                 for first, second in zip(completions, repeated)
             ],
             "audit": [{"launch_operation_id": item.audit_metadata.launch_operation_id,
                         "operation_id": item.audit_metadata.operation_id}
                        if item.audit_metadata else None for item in repeated],
             "states": [item.terminal_state.value if item.terminal_state else None
                        for item in completions]}
    _record(value)
    return json.dumps(value, sort_keys=True)

def probe(args, *, invocation):
    retained = [SubagentHandle.from_dict(item)
                for item in json.loads(handle_path.read_text(encoding="utf-8"))]
    value = {"phase": "session-denial", "task_id": invocation.task_id,
             "operation_id": invocation.operation_id,
             "session_id": invocation.session_id,
             "ctx": [ctx_ref.subagent_lifecycle.status(item).state.value
                     for item in retained],
             "invocation": [invocation.subagents.status(item).state.value
                             for item in retained],
             "listed": len(invocation.subagents.list())}
    _record(value)
    return json.dumps(value, sort_keys=True)

def nested(args, *, invocation):
    handle = invocation.subagents.launch(_request("grandchild"))
    nested_handle_path.write_text(json.dumps(handle.to_dict()), encoding="utf-8")
    return json.dumps({"nested_handle": handle.to_dict()}, sort_keys=True)

def admission_launch(args, *, invocation):
    admission_handles[:] = [
        invocation.subagents.launch(_request("admission-held-a")),
        invocation.subagents.launch(_request("admission-held-b")),
    ]
    rejection = None
    try:
        invocation.subagents.launch(_request("admission-rejected"))
    except Exception as exc:
        rejection = str(exc)
    value = {"phase": "admission-launch",
             "states": [invocation.subagents.status(handle).state.value
                        for handle in admission_handles],
             "listed": len(invocation.subagents.list()),
             "rejection": rejection}
    _record(value)
    return json.dumps(value, sort_keys=True)

def admission_cleanup(args, *, invocation):
    stopped = [invocation.subagents.stop(handle, reason="release capacity")
               for handle in admission_handles]
    terminal = [invocation.subagents.wait(handle, timeout_seconds=10)
                for handle in admission_handles]
    collected = [invocation.subagents.collect(handle) for handle in admission_handles]
    replacement = invocation.subagents.launch(SubagentLaunchRequestV2(
        api_contract_version=2,
        base=SubagentLaunchRequest(goal="contract child admission release", model="worker-model"),
        toolset_mode="exact", exact_toolsets=(), provider="lmstudio"))
    released = invocation.subagents.wait(replacement, timeout_seconds=10)
    value = {"phase": "admission-cleanup",
             "stop": [item.accepted for item in stopped],
             "held_terminal": [item.state.value for item in terminal],
             "collected_ready": [item.ready for item in collected],
             "replacement_terminal": released.state.value}
    _record(value)
    return json.dumps(value, sort_keys=True)

def matrix(args, *, invocation):
    requests = [
        ("v1-empty", SubagentLaunchRequest(
            goal="contract child v1-empty", allowed_toolsets=())),
        ("v2-inherit", SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="contract child v2-inherit"),
            toolset_mode="inherit", exact_toolsets=())),
        ("v2-exact-file", SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="contract child v2-exact-file"),
            toolset_mode="exact", exact_toolsets=("file",))),
        ("nested-parent", SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="nested launcher child", model="worker-model"),
            toolset_mode="inherit", exact_toolsets=(),
            provider="lmstudio")),
    ]
    launched = []
    for label, request in requests:
        handle = invocation.subagents.launch(request)
        launched.append((label, handle))
    terminal = {}
    for label, handle in launched:
        terminal[label] = invocation.subagents.wait(
            handle, timeout_seconds=10).state.value

    failures = {}
    canary = os.environ["CORE6_MUTATION_CANARY"]
    try:
        invocation.subagents.launch(SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="workdir mutation", working_directory=canary),
            toolset_mode="exact", exact_toolsets=()))
    except Exception as exc:
        failures["workdir"] = str(exc)
    try:
        invocation.subagents.launch(SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="ACP mutation canary", model="gpt-4.1"),
            toolset_mode="exact", exact_toolsets=(), provider="copilot-acp"))
    except Exception as exc:
        failures["copilot-acp"] = str(exc)
    try:
        invocation.subagents.launch(SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="route canary", model="worker-model"),
            toolset_mode="inherit", exact_toolsets=(),
            provider=os.environ["CORE6_PROVIDER_CANARY"]))
    except Exception as exc:
        failures["route"] = str(exc)
    try:
        invocation.subagents.launch(SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="reasoning canary", model="worker-model"),
            toolset_mode="inherit", exact_toolsets=(), provider="lmstudio",
            reasoning_effort=os.environ["CORE6_REASONING_CANARY"]))
    except Exception as exc:
        failures["reasoning"] = str(exc)
    nested_handle = SubagentHandle.from_dict(
        json.loads(nested_handle_path.read_text(encoding="utf-8")))
    nested_state = invocation.subagents.status(nested_handle).state.value
    direct_ids = [item.handle.subagent_id for item in invocation.subagents.list()]
    value = {"phase": "matrix", "task_id": invocation.task_id,
             "operation_id": invocation.operation_id,
             "session_id": invocation.session_id,
             "terminal": terminal, "failures": failures,
             "nested_state": nested_state,
             "nested_visible": nested_handle.subagent_id in direct_ids,
             "capabilities": sorted(invocation.subagents.capabilities().features)}
    _record(value)
    return json.dumps(value, sort_keys=True)

def register(ctx):
    global ctx_ref
    ctx_ref = ctx
    schema = {"description": "Hermetic lifecycle E2E",
              "parameters": {"type": "object", "properties": {}}}
    ctx.register_tool("e2e_lifecycle_launch", "plugin_lifecycle_e2e",
                      dict(schema, name="e2e_lifecycle_launch"), launch)
    ctx.register_tool("e2e_lifecycle_control", "plugin_lifecycle_e2e",
                      dict(schema, name="e2e_lifecycle_control"), control)
    ctx.register_tool("e2e_lifecycle_matrix", "plugin_lifecycle_e2e",
                      dict(schema, name="e2e_lifecycle_matrix"), matrix)
    ctx.register_tool("e2e_lifecycle_probe", "plugin_lifecycle_e2e",
                      dict(schema, name="e2e_lifecycle_probe"), probe)
    ctx.register_tool("e2e_lifecycle_nested", "plugin_lifecycle_e2e",
                      dict(schema, name="e2e_lifecycle_nested"), nested)
    ctx.register_tool("e2e_admission_launch", "plugin_lifecycle_e2e",
                      dict(schema, name="e2e_admission_launch"), admission_launch)
    ctx.register_tool("e2e_admission_cleanup", "plugin_lifecycle_e2e",
                      dict(schema, name="e2e_admission_cleanup"), admission_cleanup)
'''


def _foreign_plugin_source() -> str:
    return '''
import json, os
from pathlib import Path
from agent.subagent_lifecycle import SubagentHandle
from hermes_constants import get_hermes_home

ctx_ref = None
root = get_hermes_home()

def foreign(args, *, invocation):
    retained = [SubagentHandle.from_dict(item)
                for item in json.loads((root / "e2e-handles.json").read_text(encoding="utf-8"))]
    value = {"phase": "plugin-denial", "task_id": invocation.task_id,
             "operation_id": invocation.operation_id,
             "session_id": invocation.session_id,
             "ctx": [ctx_ref.subagent_lifecycle.status(item).state.value
                     for item in retained],
             "invocation": [invocation.subagents.status(item).state.value
                             for item in retained],
             "listed": len(invocation.subagents.list())}
    with (root / "e2e-plugin-receipts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\\n")
    return json.dumps(value, sort_keys=True)

def register(ctx):
    global ctx_ref
    ctx_ref = ctx
    ctx.register_tool("foreign_lifecycle_probe", "plugin_foreign_e2e", {
        "name": "foreign_lifecycle_probe", "description": "Cross-plugin authority probe",
        "parameters": {"type": "object", "properties": {}}}, foreign)
'''


def _sse(message: dict) -> bytes:
    chunks = [
        {"id": "e2e", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    ]
    if message.get("tool_calls"):
        for index, call in enumerate(message["tool_calls"]):
            chunks.append({"id": "e2e", "choices": [{"index": 0, "delta": {
                "tool_calls": [{"index": index, "id": call["id"], "type": "function",
                                "function": call["function"]}]}, "finish_reason": None}]})
        finish = "tool_calls"
    else:
        chunks.append({"id": "e2e", "choices": [{"index": 0, "delta": {"content": message.get("content", "")}, "finish_reason": None}]})
        finish = "stop"
    chunks.append({"id": "e2e", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks) + b"data: [DONE]\n\n"


class _Provider(BaseHTTPRequestHandler):
    requests: list[dict] = []
    request_lock = threading.Lock()
    child_barrier = threading.Condition(request_lock)
    delayed_children = 0
    hold_children = False
    canary_path = ""
    admission_step = 0

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/api/v1/models"):
            body = json.dumps({"models": [{
                "key": "worker-model", "type": "llm",
                "loaded_instances": [{"config": {"context_length": 131072}}],
                "capabilities": {"reasoning": {
                    "allowed_options": ["off", "low", "medium", "high"]
                }},
            }]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"data": [
                {"id": "parent-model"}, {"id": "worker-model"}
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        with type(self).request_lock:
            type(self).requests.append(request)
        messages = request.get("messages") or []
        last_user = next((str(item.get("content", "")) for item in reversed(messages)
                          if item.get("role") == "user"), "")
        has_tool_result = bool(messages and messages[-1].get("role") == "tool")
        tools = request.get("tools") or []
        called_tools = [
            call.get("function", {}).get("name")
            for item in messages for call in item.get("tool_calls") or []
        ]
        if "contract child" in last_user:
            message = {"role": "assistant", "content": "contract child complete"}
        elif "delayed read-only" in last_user:
            with type(self).child_barrier:
                type(self).delayed_children += 1
                type(self).child_barrier.notify_all()
                while type(self).hold_children:
                    type(self).child_barrier.wait(timeout=0.5)
            time.sleep(float(os.environ.get("CORE6_CHILD_DELAY", "1.0")))
            child_has_tool_result = any(
                item.get("role") == "tool" for item in messages
            )
            if not child_has_tool_result:
                canary = type(self).canary_path or os.environ.get(
                    "CORE6_MUTATION_CANARY", "/forbidden/core6-canary"
                )
                message = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "child-mutation-canary", "type": "function",
                    "function": {"name": "write_file", "arguments": json.dumps({
                        "path": canary, "content": "must-not-exist"
                    })},
                }]}
            else:
                message = {"role": "assistant", "content": "child completed read-only"}
        elif "nested launcher child" in last_user and not has_tool_result:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "nested-launch", "type": "function",
                "function": {"name": "e2e_lifecycle_nested", "arguments": "{}"}}]}
        elif (
            "TURN_A" in last_user
            and "TURN_ADMISSION" not in last_user
            and not has_tool_result
        ):
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "turn-a-launch", "type": "function",
                "function": {"name": "e2e_lifecycle_launch", "arguments": "{}"}}]}
        elif "TURN_B" in last_user and not has_tool_result:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "turn-b-control", "type": "function",
                "function": {"name": "e2e_lifecycle_control", "arguments": "{}"}}]}
        elif "TURN_MATRIX" in last_user and not has_tool_result:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "turn-matrix", "type": "function",
                "function": {"name": "e2e_lifecycle_matrix", "arguments": "{}"}}]}
        elif "TURN_FOREIGN" in last_user and not has_tool_result:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "turn-foreign", "type": "function",
                "function": {"name": "foreign_lifecycle_probe", "arguments": "{}"}}]}
        elif "TURN_DENIED_SESSION" in last_user and not has_tool_result:
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "turn-denied-session", "type": "function",
                "function": {"name": "e2e_lifecycle_probe", "arguments": "{}"}}]}
        elif "TURN_ADMISSION" in last_user:
            admission_tools_ready = any(
                item.get("function", {}).get("name") == "e2e_admission_launch"
                for item in tools
            )
            if not admission_tools_ready:
                message = {"role": "assistant", "content": "admission bootstrap"}
                name = None
                step = -1
            else:
                step = type(self).admission_step
                type(self).admission_step += 1
            if step == 0:
                name, arguments = "e2e_admission_launch", {}
            elif step == 1:
                name, arguments = "delegate_task", {
                    "goal": "contract child admission delegate fallback"
                }
            elif step == 2:
                name, arguments = "delegate_task", {"tasks": [
                    {"goal": "contract child admission batch first"},
                    {"goal": "contract child admission batch second"},
                ]}
            elif step == 3:
                name, arguments = "cronjob", {
                    "action": "create", "name": "core6-admission-job",
                    "prompt": "contract child cron admission", "schedule": "2099-01-01T00:00:00"
                }
            elif step == 4:
                name, arguments = "cronjob", {
                    "action": "run", "job_id": "core6-admission-job"
                }
            elif step == 5:
                name, arguments = "e2e_admission_cleanup", {}
            else:
                name = None
            if name:
                message = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "admission-" + str(step), "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}]}
            else:
                message = {"role": "assistant", "content": "admission complete"}
        else:
            message = {"role": "assistant", "content": "gateway turn complete"}
        body = _sse(message) if request.get("stream") else json.dumps({
            "id": "e2e", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if request.get("stream") else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


def _run_host(root: Path, phase: str = "both") -> int:
    _Provider.requests = []
    _Provider.admission_step = 0
    external_loopback = os.environ.get("CORE6_EXTERNAL_LOOPBACK_URL", "").rstrip("/")
    server = None
    thread = None
    if external_loopback:
        base_url = external_loopback
    else:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
        port = int(server.server_address[1])
        thread = threading.Thread(target=server.serve_forever, name="core6-loopback", daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}/v1"
    hermes_home = Path(os.environ["HERMES_HOME"])
    config = _isolated_config(base_url)
    if phase == "admission":
        config["delegation"]["max_concurrent_children"] = 2
    configured_toolsets = os.environ.get("CORE6_PARENT_TOOLSETS", "").strip()
    if configured_toolsets:
        config["platform_toolsets"]["cli"] = [
            item.strip() for item in configured_toolsets.split(",") if item.strip()
        ]
    config_path = hermes_home / "config.yaml"
    if not config_path.exists():
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env_path = hermes_home / ".env"
    if not env_path.exists():
        env_path.write_text(
            f"LM_BASE_URL={base_url}\n"
            "LM_API_KEY=synthetic-lm-key\n"
            f"HERMES_COPILOT_ACP_COMMAND={os.environ['HERMES_COPILOT_ACP_COMMAND']}\n"
            f"HERMES_COPILOT_ACP_ARGS={os.environ['HERMES_COPILOT_ACP_ARGS']}\n",
            encoding="utf-8",
        )
    other_home = root / "other-hermes"
    if phase == "profile":
        (other_home / "config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        (other_home / ".env").write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
    liveness_socket = None
    if phase == "a":
        liveness_socket = socket.socket()
        liveness_socket.bind(("127.0.0.1", 0))
        liveness_socket.listen(1)
        (root / "host-a-liveness.json").write_text(
            json.dumps({"port": int(liveness_socket.getsockname()[1])}), encoding="utf-8"
        )

    # Core imports intentionally occur only after the supervisor supplied the
    # isolated HOME/HERMES_HOME/TMPDIR and the loopback-backed config exists.
    # Several production config readers initialize during import; writing the
    # config afterward would test a stale startup snapshot rather than the
    # installed host path.
    import asyncio
    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    import gateway.run as gateway_run_module
    from gateway.session import SessionSource
    from hermes_cli.plugins import discover_plugins, get_plugin_manager, get_plugin_toolsets
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override


    session_files_before = len(list((hermes_home / "sessions").rglob("*")))
    discover_plugins(force=True)
    manager = get_plugin_manager()
    runner = GatewayRunner(GatewayConfig(sessions_dir=hermes_home / "sessions"))
    source = SessionSource(platform=Platform.LOCAL, chat_id="core6", chat_type="dm", user_id="owner")
    foreign_source = SessionSource(
        platform=Platform.LOCAL, chat_id="core6-other", chat_type="dm", user_id="owner"
    )
    from hermes_cli.config import load_config
    from model_tools import get_tool_definitions
    selected_toolsets = runner._resolve_enabled_toolsets_for_source(
        load_config(), source, "cli"
    )
    selected_schemas = get_tool_definitions(selected_toolsets, quiet_mode=True)

    other_receipts = []
    other_manager = None

    async def turns():
        nonlocal other_manager, other_receipts
        responses = []
        if phase in {"a", "both"}:
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_A", source=source, message_id="message-a", internal=True)
            ))
        if phase in {"b", "both"}:
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_B", source=source, message_id="message-b", internal=True)
            ))
        if phase == "matrix":
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_MATRIX", source=source, message_id="message-matrix", internal=True)
            ))
        if phase == "authority":
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_A", source=source, message_id="authority-a", internal=True)
            ))
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_FOREIGN", source=source, message_id="authority-plugin", internal=True)
            ))
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_DENIED_SESSION", source=foreign_source,
                             message_id="authority-session", internal=True)
            ))
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_B", source=source, message_id="authority-cleanup", internal=True)
            ))
        if phase == "profile":
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_A", source=source, message_id="profile-owner", internal=True)
            ))
            shutil.copy2(handle_path := hermes_home / "e2e-handles.json",
                         other_home / handle_path.name)
            shutil.copytree(
                hermes_home / "sessions", other_home / "sessions", dirs_exist_ok=True
            )
            token = set_hermes_home_override(other_home)
            try:
                discover_plugins(force=True)
                other_manager = get_plugin_manager()
                other_runner = GatewayRunner(
                    GatewayConfig(sessions_dir=other_home / "sessions")
                )
                responses.append(await other_runner._handle_message(
                    MessageEvent(text="TURN_DENIED_SESSION", source=source,
                                 message_id="profile-denial", internal=True)
                ))
                other_receipt_path = other_home / "e2e-plugin-receipts.jsonl"
                other_receipts = [
                    json.loads(line) for line in other_receipt_path.read_text().splitlines()
                ]
            finally:
                reset_hermes_home_override(token)
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_B", source=source,
                             message_id="profile-cleanup", internal=True)
            ))
        if phase == "admission":
            responses.append(await runner._handle_message(
                MessageEvent(text="TURN_ADMISSION", source=source,
                             message_id="admission", internal=True)
            ))
        return responses

    try:
        responses = asyncio.run(turns())
        receipt_file = hermes_home / "e2e-plugin-receipts.jsonl"
        receipts = ([json.loads(line) for line in receipt_file.read_text().splitlines()]
                    if receipt_file.exists() else [])
        child_posts = [
            request for request in _Provider.requests
            if any("delayed read-only" in str(message.get("content", ""))
                   for message in request.get("messages") or [])
        ]
        contract_posts = [
            request for request in _Provider.requests
            if any("contract child" in str(message.get("content", ""))
                   for message in request.get("messages") or [])
        ]
        contract_schemas = {}
        for request in contract_posts:
            prompt = " ".join(
                str(message.get("content", ""))
                for message in request.get("messages") or []
                if message.get("role") == "user"
            )
            for label in ("v1-empty", "v2-inherit", "v2-exact-file"):
                if label in prompt:
                    contract_schemas[label] = sorted(
                        item.get("function", {}).get("name")
                        for item in request.get("tools") or []
                    )
        parent_posts = [
            request for request in _Provider.requests
            if any(item.get("function", {}).get("name") == "e2e_lifecycle_launch"
                   for item in request.get("tools") or [])
        ]
        result = {"responses": responses, "receipts": receipts,
                  "other_receipts": other_receipts,
                  "post_count": len(_Provider.requests),
                  "child_tool_counts": [len(item.get("tools") or []) for item in child_posts],
                  "manager_scope": manager.scope_key, "session_files": len(list((hermes_home / "sessions").rglob("*"))),
                  "session_files_before": session_files_before,
                  "plugin_toolsets": get_plugin_toolsets(),
                  "request_tools": [[item.get("function", {}).get("name") for item in request.get("tools") or []]
                                    for request in _Provider.requests],
                  "tool_results": [str(message.get("content", ""))[:240]
                                   for request in _Provider.requests
                                   for message in request.get("messages") or []
                                   if message.get("role") == "tool"],
                  "tool_results_by_call": {
                      str(message.get("tool_call_id")): str(message.get("content", ""))
                      for request in _Provider.requests
                      for message in request.get("messages") or []
                      if message.get("role") == "tool" and message.get("tool_call_id")
                  },
                  "has_e2e_toolset": any(item[0] == "plugin_lifecycle_e2e" for item in get_plugin_toolsets()),
                  "has_e2e_schema": any(
                      item.get("function", {}).get("name") == "e2e_lifecycle_launch"
                      for request in _Provider.requests for item in request.get("tools") or []),
                  "selected_toolsets": selected_toolsets,
                  "selected_schemas": [item.get("function", {}).get("name")
                                       for item in selected_schemas],
                  "parent_prompt_stable": bool(parent_posts) and len({
                      json.dumps((request.get("messages") or [{}])[0], sort_keys=True)
                      for request in parent_posts
                  }) == 1,
                  "parent_schema_stable": bool(parent_posts) and len({
                      json.dumps(request.get("tools") or [], sort_keys=True)
                      for request in parent_posts
                  }) == 1,
                  "parent_models": sorted({request.get("model") for request in parent_posts}),
                  "child_models": sorted({request.get("model") for request in child_posts}),
                  "child_reasoning_efforts": sorted({
                      request.get("reasoning_effort") for request in child_posts
                      if request.get("reasoning_effort") is not None
                  }),
                  "contract_schemas": contract_schemas}
        candidate_root = Path(__file__).resolve().parents[2]
        result["candidate_module"] = str(Path(gateway_run_module.__file__).resolve())
        result["candidate_commit"] = subprocess.run(
            ["git", "-C", str(candidate_root), "rev-parse", "HEAD"],
            check=True, text=True, capture_output=True, timeout=10,
        ).stdout.strip()
        serialized = json.dumps(result, sort_keys=True)
        (root / f"host-{phase}-receipt.json").write_text(serialized, encoding="utf-8")
        print(serialized, flush=True)
        if phase == "a":
            while True:
                time.sleep(1)
        return 0
    finally:
        if other_manager is not None:
            other_manager.unload()
        manager.unload()
        if server is not None and thread is not None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            assert not thread.is_alive()
        if liveness_socket is not None:
            liveness_socket.close()


def _prepare_isolated_root(tmp_path: Path, name: str = "core6"):
    root = tmp_path / name
    home, hermes_home, temp_dir = root / "home", root / "hermes", root / "tmp"
    plugin = hermes_home / "plugins" / "lifecycle-e2e"
    foreign_plugin = hermes_home / "plugins" / "foreign-e2e"
    repo, worker = root / "repo", root / "worker"
    for directory in (home, temp_dir, plugin, foreign_plugin):
        directory.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "lifecycle-e2e", "version": "1.0.0",
        "provides_tools": ["e2e_lifecycle_launch", "e2e_lifecycle_control",
                           "e2e_lifecycle_matrix", "e2e_lifecycle_probe",
                           "e2e_lifecycle_nested", "e2e_admission_launch",
                           "e2e_admission_cleanup"],
    }), encoding="utf-8")
    (plugin / "__init__.py").write_text(_plugin_source(), encoding="utf-8")
    (foreign_plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "foreign-e2e", "version": "1.0.0",
        "provides_tools": ["foreign_lifecycle_probe"],
    }), encoding="utf-8")
    (foreign_plugin / "__init__.py").write_text(_foreign_plugin_source(), encoding="utf-8")
    other_plugins = root / "other-hermes" / "plugins"
    shutil.copytree(hermes_home / "plugins", other_plugins)
    subprocess.run(["git", "init", "-q", "--initial-branch=main", str(repo)], check=True, timeout=10)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Core6 E2E"], check=True, timeout=10)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "core6@example.invalid"], check=True, timeout=10)
    (repo / "seed").write_text("isolated\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "seed"], check=True, timeout=10)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "seed"], check=True, timeout=10)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "-b", "worker", str(worker)], check=True, timeout=10)
    env = {
        key: value for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in (
            "API_KEY", "TOKEN", "SECRET", "PASSWORD", "PROXY"
        ))
    }
    env.update({"HOME": str(home), "HERMES_HOME": str(hermes_home), "TMPDIR": str(temp_dir),
                "PYTHONDONTWRITEBYTECODE": "1", "NO_COLOR": "1", "NO_PROXY": "127.0.0.1,localhost",
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                "HERMES_GATEWAY_SESSION": "1"})
    env["CORE6_MUTATION_CANARY"] = str(worker / "must-not-be-written")
    env["CORE6_PROVIDER_CANARY"] = "provider-key-url-env-path-command-canary"
    env["CORE6_REASONING_CANARY"] = "reasoning-key-url-env-path-command-canary"
    fake_bin = root / "bin"
    fake_bin.mkdir()
    fake_copilot = fake_bin / "copilot"
    fake_copilot.write_text(
        f"#!/bin/sh\ntouch {json.dumps(str(root / 'copilot-executed'))}\nexit 97\n",
        encoding="utf-8",
    )
    fake_copilot.chmod(0o700)
    env["HERMES_COPILOT_ACP_COMMAND"] = str(fake_copilot)
    env["HERMES_COPILOT_ACP_ARGS"] = (
        "--acp --stdio writeTextFile " + env["CORE6_MUTATION_CANARY"]
    )
    return root, repo, worker, env


def _remove_isolated_worktree(root: Path, repo: Path, worker: Path):
    subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(worker)], timeout=10)
    assert not worker.exists()
    assert not list(root.rglob("*.sock"))
    assert not list(root.rglob("__pycache__"))
    worktrees = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        check=True, text=True, capture_output=True, timeout=10,
    ).stdout
    assert str(worker) not in worktrees
    assert subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True, text=True, capture_output=True, timeout=10,
    ).stdout == ""


def test_external_real_gateway_turns_have_distinct_operations_and_zero_child_tools(tmp_path):
    candidate_root = Path(__file__).resolve().parents[2]
    expected_candidate_commit = subprocess.run(
        ["git", "-C", str(candidate_root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True, timeout=10,
    ).stdout.strip()
    root, repo, worker, env = _prepare_isolated_root(tmp_path)
    completed = None
    try:
        completed = subprocess.run([sys.executable, __file__, "--host", str(root)], cwd=repo, env=env,
                                   text=True, capture_output=True, timeout=_HOST_TIMEOUT)
        assert completed.returncode == 0, completed.stderr[-4000:]
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        assert [item["phase"] for item in receipt["receipts"]] == ["launch", "control"], (
            json.dumps(receipt) + "\n" + completed.stderr[-8000:]
        )
        assert len({item["task_id"] for item in receipt["receipts"]}) == 1
        assert len({item["session_id"] for item in receipt["receipts"]}) == 1
        assert receipt["receipts"][0]["task_id"] == receipt["receipts"][0]["session_id"]
        operation_ids = [item["operation_id"] for item in receipt["receipts"]]
        assert len(set(operation_ids)) == 2
        assert not set(operation_ids) & {"message-a", "message-b"}
        assert receipt["receipts"][1]["ready"] == [True, True]
        assert receipt["receipts"][1]["stable"] == [True, True]
        assert receipt["receipts"][1]["before"] == {
            "ctx": ["RUNNING", "RUNNING"],
            "invocation": ["RUNNING", "RUNNING"],
        }
        assert receipt["receipts"][1]["status_before"] == ["RUNNING", "RUNNING"]
        assert receipt["receipts"][1]["steer"] == "QUEUED"
        assert receipt["receipts"][1]["stop"] is True
        assert set(receipt["receipts"][1]["states"]) <= {"SUCCEEDED", "CANCELLED"}
        assert all(receipt["receipts"][1]["events"])
        launch_audit = receipt["receipts"][0]["audit"]
        control_audit = receipt["receipts"][1]["audit"]
        assert all(item["launch_operation_id"] == operation_ids[0] for item in launch_audit)
        assert all(item["launch_operation_id"] == operation_ids[0] for item in control_audit)
        assert all(item["operation_id"] == operation_ids[1] for item in control_audit)
        assert receipt["child_tool_counts"] and set(receipt["child_tool_counts"]) == {0}
        assert not Path(env["CORE6_MUTATION_CANARY"]).exists()
        assert any("write_file" in item for item in receipt["tool_results"]), receipt["tool_results"]
        assert receipt["post_count"] >= 6
        assert receipt["responses"] == ["gateway turn complete", "gateway turn complete"]
        assert receipt["has_e2e_toolset"] and receipt["has_e2e_schema"]
        assert receipt["selected_toolsets"] == [
            "kanban", "plugin_foreign_e2e", "plugin_lifecycle_e2e"
        ]
        assert receipt["selected_schemas"] == [
            "e2e_admission_cleanup", "e2e_admission_launch",
            "e2e_lifecycle_control", "e2e_lifecycle_launch", "e2e_lifecycle_matrix",
            "e2e_lifecycle_nested", "e2e_lifecycle_probe", "foreign_lifecycle_probe"
        ]
        assert Path(receipt["candidate_module"]).is_relative_to(
            Path(__file__).resolve().parents[2]
        )
        assert receipt["candidate_commit"] == expected_candidate_commit
        assert receipt["parent_prompt_stable"] and receipt["parent_schema_stable"]
        assert receipt["parent_models"] == ["parent-model"]
        assert receipt["child_models"] == ["worker-model"]
        assert receipt["child_reasoning_efforts"] == ["high"]
    finally:
        _remove_isolated_worktree(root, repo, worker)
        if completed is not None:
            assert completed.returncode is not None


def test_isolated_host_restart_loses_inflight_native_child_and_releases_process_group(tmp_path):
    root, repo, worker, env = _prepare_isolated_root(tmp_path, "core6-restart")
    _Provider.requests = []
    _Provider.delayed_children = 0
    _Provider.hold_children = True
    _Provider.canary_path = env["CORE6_MUTATION_CANARY"]
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    provider_port = int(provider.server_address[1])
    provider_thread = threading.Thread(
        target=provider.serve_forever, name="core6-supervisor-loopback", daemon=True
    )
    provider_thread.start()
    base_url = f"http://127.0.0.1:{provider_port}/v1"
    config_path = root / "hermes" / "config.yaml"
    config_bytes = yaml.safe_dump(_isolated_config(base_url)).encode()
    config_path.write_bytes(config_bytes)
    env["CORE6_EXTERNAL_LOOPBACK_URL"] = base_url
    first = None
    try:
        first = subprocess.Popen(
            [sys.executable, __file__, "--host", str(root), "a"],
            cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        ready = root / "host-a-receipt.json"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not ready.exists():
            assert first.poll() is None, (first.stderr.read() if first.stderr else "")[-4000:]
            time.sleep(0.05)
        assert ready.exists(), "turn A never flushed its launch receipt"
        launched = json.loads(ready.read_text(encoding="utf-8"))
        assert [item["phase"] for item in launched["receipts"]] == ["launch"]
        assert launched["receipts"][0]["states"] == ["RUNNING", "RUNNING"]
        with _Provider.child_barrier:
            deadline = time.monotonic() + 10
            while _Provider.delayed_children < 2 and time.monotonic() < deadline:
                _Provider.child_barrier.wait(timeout=0.1)
            assert _Provider.delayed_children == 2
        liveness_port = json.loads(
            (root / "host-a-liveness.json").read_text(encoding="utf-8")
        )["port"]
        with socket.create_connection(("127.0.0.1", liveness_port), timeout=1):
            pass

        process_group = os.getpgid(first.pid)
        os.killpg(process_group, 15)
        first.wait(timeout=10)
        assert first.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(process_group, 0)
        closed = socket.socket()
        try:
            assert closed.connect_ex(("127.0.0.1", liveness_port)) != 0
        finally:
            closed.close()
        persisted_config = config_path.read_bytes()
        persisted_values = yaml.safe_load(persisted_config)
        assert persisted_values["model"] == _isolated_config(base_url)["model"]
        assert persisted_values["custom_providers"] == _isolated_config(base_url)["custom_providers"]

        second = subprocess.run(
            [sys.executable, __file__, "--host", str(root), "b"],
            cwd=repo, env=env, text=True, capture_output=True, timeout=_HOST_TIMEOUT,
        )
        assert second.returncode == 0, second.stderr[-4000:]
        assert config_path.read_bytes() == persisted_config
        restarted = json.loads(second.stdout.strip().splitlines()[-1])
        assert [item["phase"] for item in restarted["receipts"]] == ["launch", "control"]
        launch, control = restarted["receipts"]
        assert launch["session_id"] == control["session_id"]
        assert launch["task_id"] == control["task_id"] == launch["session_id"]
        assert launch["operation_id"] != control["operation_id"]
        assert restarted["session_files_before"] >= 1
        assert control["ready"] == [False, False]
        assert control["diagnostics"] == ["UNKNOWN_HANDLE", "UNKNOWN_HANDLE"]
        assert not Path(env["CORE6_MUTATION_CANARY"]).exists()
    finally:
        with _Provider.child_barrier:
            _Provider.hold_children = False
            _Provider.child_barrier.notify_all()
        _Provider.canary_path = ""
        if first is not None and first.poll() is None:
            try:
                os.killpg(os.getpgid(first.pid), 15)
            except ProcessLookupError:
                pass
            first.wait(timeout=10)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
        assert not provider_thread.is_alive()
        _remove_isolated_worktree(root, repo, worker)


def test_real_gateway_v1_v2_toolsets_workdir_and_acp_transport_fail_closed(tmp_path):
    root, repo, worker, env = _prepare_isolated_root(tmp_path, "core6-contracts")
    completed = None
    try:
        # Widen the parent to the built-in file toolset so exact-nonempty can
        # prove a real subset construction. Host still owns the port-0 route.
        env["CORE6_PARENT_TOOLSETS"] = "file,plugin_lifecycle_e2e"
        completed = subprocess.run(
            [sys.executable, __file__, "--host", str(root), "matrix"],
            cwd=repo, env=env, text=True, capture_output=True, timeout=_HOST_TIMEOUT,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        assert receipt["receipts"], json.dumps(receipt, sort_keys=True)
        matrix = receipt["receipts"][-1]
        assert matrix["phase"] == "matrix"
        assert set(matrix["terminal"].values()) == {"SUCCEEDED"}
        assert matrix["failures"] == {
            "workdir": (
                "working_directory is not supported because Hermes delegates "
                "use isolated task environments."
            ),
            "copilot-acp": "Native read-only transport is unavailable.",
            "route": "Requested provider/model route is unavailable.",
            "reasoning": (
                "reasoning_effort must be false, 'none', or a supported "
                "effort identifier."
            ),
        }
        schemas = receipt["contract_schemas"]
        assert schemas["v1-empty"] == schemas["v2-inherit"]
        assert "e2e_lifecycle_matrix" in schemas["v1-empty"]
        assert "write_file" in schemas["v1-empty"]
        assert "write_file" in schemas["v2-exact-file"]
        assert "e2e_lifecycle_matrix" not in schemas["v2-exact-file"]
        assert set(schemas["v2-exact-file"]) < set(schemas["v1-empty"])
        assert "native_read_only_transport_gate" in matrix["capabilities"]
        assert matrix["nested_state"] == "UNKNOWN"
        assert matrix["nested_visible"] is False
        assert not Path(env["CORE6_MUTATION_CANARY"]).exists()
        assert not (root / "copilot-executed").exists()
        serialized = json.dumps(matrix, sort_keys=True)
        for canary in (
            env["CORE6_MUTATION_CANARY"], env["CORE6_PROVIDER_CANARY"],
            env["CORE6_REASONING_CANARY"], "synthetic-lm-key",
            "HERMES_COPILOT_ACP_ARGS", "writeTextFile",
        ):
            assert canary not in serialized
    finally:
        _remove_isolated_worktree(root, repo, worker)


def test_real_gateway_cross_plugin_and_session_authority_fail_closed(tmp_path):
    root, repo, worker, env = _prepare_isolated_root(tmp_path, "core6-authority")
    completed = None
    try:
        env["CORE6_PARENT_TOOLSETS"] = (
            "plugin_lifecycle_e2e,plugin_foreign_e2e"
        )
        completed = subprocess.run(
            [sys.executable, __file__, "--host", str(root), "authority"],
            cwd=repo, env=env, text=True, capture_output=True, timeout=_HOST_TIMEOUT,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        phases = {item["phase"]: item for item in receipt["receipts"]}
        assert set(phases) == {"launch", "plugin-denial", "session-denial", "control"}
        for phase in ("plugin-denial", "session-denial"):
            assert phases[phase]["ctx"] == ["UNKNOWN", "UNKNOWN"]
            assert phases[phase]["invocation"] == ["UNKNOWN", "UNKNOWN"]
            assert phases[phase]["listed"] == 0
        assert phases["plugin-denial"]["session_id"] == phases["launch"]["session_id"]
        assert phases["session-denial"]["session_id"] != phases["launch"]["session_id"]
        assert phases["control"]["ready"] == [True, True]
        assert phases["control"]["stable"] == [True, True]
        assert not Path(env["CORE6_MUTATION_CANARY"]).exists()
    finally:
        _remove_isolated_worktree(root, repo, worker)


def test_real_gateway_cross_profile_and_manager_authority_fail_closed(tmp_path):
    root, repo, worker, env = _prepare_isolated_root(tmp_path, "core6-profile")
    completed = None
    try:
        completed = subprocess.run(
            [sys.executable, __file__, "--host", str(root), "profile"],
            cwd=repo, env=env, text=True, capture_output=True, timeout=_HOST_TIMEOUT,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        assert len(receipt["other_receipts"]) == 1
        denial = receipt["other_receipts"][0]
        assert denial["phase"] == "session-denial"
        assert denial["ctx"] == denial["invocation"] == ["UNKNOWN", "UNKNOWN"]
        assert denial["listed"] == 0
        assert denial["session_id"] == receipt["receipts"][0]["session_id"]
        assert receipt["manager_scope"] != str((root / "other-hermes").resolve())
        assert receipt["receipts"][-1]["phase"] == "control"
        assert receipt["receipts"][-1]["ready"] == [True, True]
    finally:
        _remove_isolated_worktree(root, repo, worker)


def test_real_gateway_shared_admission_preserves_delegate_cron_and_batch_contracts(tmp_path):
    root, repo, worker, env = _prepare_isolated_root(tmp_path, "core6-admission")
    completed = None
    try:
        env["CORE6_PARENT_TOOLSETS"] = (
            "delegation,cronjob,plugin_lifecycle_e2e"
        )
        env["CORE6_CHILD_DELAY"] = "10"
        completed = subprocess.run(
            [sys.executable, __file__, "--host", str(root), "admission"],
            cwd=repo, env=env, text=True, capture_output=True, timeout=_HOST_TIMEOUT,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        receipt = json.loads(completed.stdout.strip().splitlines()[-1])
        phases = {item["phase"]: item for item in receipt["receipts"]}
        assert "admission-launch" in phases, json.dumps(receipt, sort_keys=True)
        assert phases["admission-launch"]["states"] == ["RUNNING", "RUNNING"]
        assert phases["admission-launch"]["rejection"] == "CAPACITY_REACHED"
        assert phases["admission-launch"]["listed"] == 2
        assert len(receipt["child_tool_counts"]) == 2
        assert phases["admission-cleanup"]["stop"] == [True, True]
        assert set(phases["admission-cleanup"]["held_terminal"]) <= {
            "CANCELLED", "INTERRUPTED"
        }
        assert phases["admission-cleanup"]["collected_ready"] == [True, True]
        assert phases["admission-cleanup"]["replacement_terminal"] == "SUCCEEDED"
        results = receipt["tool_results_by_call"]
        single = json.loads(results["admission-1"])
        batch = json.loads(results["admission-2"])
        cron_create = json.loads(results["admission-3"])
        cron_run = json.loads(results["admission-4"])
        assert len(single["results"]) == 1
        assert single["results"][0]["status"] == "completed"
        assert len(batch["results"]) == 2
        assert {item["status"] for item in batch["results"]} == {"completed"}
        assert cron_create["success"] is True
        assert cron_create["name"] == "core6-admission-job"
        assert cron_run["success"] is True
        assert cron_run["job"]["name"] == "core6-admission-job"
        assert "delegation_id" not in cron_run
        assert "admission complete" in receipt["responses"]
        assert not Path(env["CORE6_MUTATION_CANARY"]).exists()
    finally:
        _remove_isolated_worktree(root, repo, worker)


if __name__ == "__main__" and len(sys.argv) in {3, 4} and sys.argv[1] == "--host":
    raise SystemExit(_run_host(Path(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else "both"))
