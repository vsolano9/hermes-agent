"""Behavior tests for the model-free Codex Computer Use broker."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import socket
import threading
import time
import uuid

import pytest
from websockets.sync.server import unix_serve

from agent.transports.codex_app_server import CodexAppServerTransportError
from tools.mcp_pinned_surfaces import CODEX_CUA_TOOL_NAMES, expected_app_server_tools


def _catalog_row() -> dict:
    return {
        "name": "computer-use",
        "pluginId": "computer-use@openai-bundled",
        "authStatus": "unsupported",
        "resourceTemplates": [],
        "resources": [],
        "tools": expected_app_server_tools(),
    }


class FakeClient:
    def __init__(
        self, *, catalog=None, tool_result=None, failure=None, delete_failure=None,
        close_failure=None,
    ) -> None:
        self.catalog = catalog if catalog is not None else [_catalog_row()]
        self.tool_result = tool_result or {
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
            "structuredContent": {"windows": []},
        }
        self.failure = failure
        self.delete_failure = delete_failure
        self.close_failure = close_failure
        self.requests: list[tuple[str, dict]] = []
        self.closed = False

    def initialize(self, **kwargs):
        return {
            "userAgent": "codex-cli/0.149.0-alpha.4.3",
            "codexHome": "/tmp/codex-home",
            "platformFamily": "unix",
            "platformOs": "macos",
        }

    def request(self, method, params=None, timeout=30.0):
        params = params or {}
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "thread-1", "ephemeral": True, "path": None}}
        if method == "mcpServerStatus/list":
            return {"data": self.catalog, "nextCursor": None}
        if method == "mcpServer/tool/call":
            if self.failure is not None:
                raise self.failure
            return deepcopy(self.tool_result)
        if method == "thread/delete":
            if self.delete_failure is not None:
                raise self.delete_failure
            return {}
        raise AssertionError(f"unexpected request: {method}")

    def close(self):
        self.closed = True
        if self.close_failure is not None:
            raise self.close_failure


class FakeDaemon:
    def __init__(self, version="0.149.0-alpha.4.3") -> None:
        self.version = version
        self.calls = []

    def ensure_ready(self, verified, *, deadline=None):
        assert deadline is None or deadline > time.monotonic()
        self.calls.append(verified)
        return type("Ready", (), {
            "socket_path": "/tmp/app-server-control.sock",
            "version": self.version,
        })()


def _broker(client, daemon=None):
    from agent.transports.codex_cua_broker import CodexCUABroker, VerifiedCodex

    verified = VerifiedCodex(
        path="/Applications/ChatGPT.app/Contents/Resources/codex",
        version="0.149.0-alpha.4.3",
        path_snapshot=(),
    )
    return CodexCUABroker(
        binary_resolver=lambda **_kwargs: verified,
        daemon_controller=daemon or FakeDaemon(),
        client_factory=lambda _socket, _open_timeout: client,
        cwd="/tmp",
    )


def test_model_free_call_attests_then_calls_and_always_deletes_thread() -> None:
    client = FakeClient()

    result = _broker(client).call("list_apps", {}, timeout=2.0)

    assert result["structuredContent"] == {"windows": []}
    assert [method for method, _params in client.requests] == [
        "thread/start",
        "mcpServerStatus/list",
        "mcpServer/tool/call",
        "thread/delete",
    ]
    assert client.requests[0][1] == {"cwd": "/tmp", "ephemeral": True}
    assert client.requests[1][1]["threadId"] == "thread-1"
    assert client.requests[1][1]["detail"] == "full"
    assert client.requests[2][1] == {
        "server": "computer-use",
        "threadId": "thread-1",
        "tool": "list_apps",
        "arguments": {},
    }
    assert all(method != "turn/start" for method, _params in client.requests)
    assert client.closed is True


def test_catalog_plugin_or_tool_drift_fails_before_tool_call() -> None:
    from agent.transports.codex_cua_broker import CodexCUACatalogDrift

    cases = []
    wrong_plugin = _catalog_row()
    wrong_plugin["pluginId"] = "computer-use@attacker"
    cases.append([wrong_plugin])
    extra_tool = _catalog_row()
    extra_tool["tools"]["shell"] = {"name": "shell"}
    cases.append([extra_tool])
    changed_schema = _catalog_row()
    changed_schema["tools"]["click"]["inputSchema"] = {"type": "object"}
    cases.append([changed_schema])
    unknown_status = _catalog_row()
    unknown_status["runtimeStatus"] = "connected"
    cases.append([unknown_status])
    changed_server_identity = _catalog_row()
    changed_server_identity["serverInfo"] = {
        "name": "computer-use",
        "version": "future",
    }
    cases.append([changed_server_identity])
    duplicate_plugin = _catalog_row()
    duplicate_plugin["name"] = "attacker-alias"
    cases.append([_catalog_row(), duplicate_plugin])

    for catalog in cases:
        client = FakeClient(catalog=catalog)
        with pytest.raises(CodexCUACatalogDrift):
            _broker(client).call("list_apps", {}, timeout=2.0)
        assert all(
            method != "mcpServer/tool/call" for method, _params in client.requests
        )
        assert client.requests[-1][0] == "thread/delete"
        assert client.closed is True


def test_repeating_catalog_cursor_fails_closed_without_spinning() -> None:
    from agent.transports.codex_cua_broker import CodexCUACatalogDrift

    class RepeatingCursorClient(FakeClient):
        def request(self, method, params=None, timeout=30.0):
            if method != "mcpServerStatus/list":
                return super().request(method, params, timeout)
            params = params or {}
            self.requests.append((method, params))
            if sum(name == method for name, _ in self.requests) > 2:
                raise AssertionError("repeating cursor must be rejected")
            return {
                "data": [_catalog_row()] if params.get("cursor") is None else [],
                "nextCursor": "same-cursor",
            }

    client = RepeatingCursorClient()
    with pytest.raises(CodexCUACatalogDrift, match="cursor"):
        _broker(client).call("list_apps", {}, timeout=2.0)
    assert client.requests[-1][0] == "thread/delete"


@pytest.mark.parametrize("mode", ["unique-cursors", "oversized-page"])
def test_catalog_pagination_is_strictly_bounded(mode) -> None:
    from agent.transports.codex_cua_broker import CodexCUACatalogDrift

    class UnboundedCatalogClient(FakeClient):
        def request(self, method, params=None, timeout=30.0):
            if method != "mcpServerStatus/list":
                return super().request(method, params, timeout)
            params = params or {}
            self.requests.append((method, params))
            page_number = sum(name == method for name, _ in self.requests)
            if page_number > 20:
                raise AssertionError("catalog pagination must be bounded")
            if mode == "oversized-page":
                return {"data": [_catalog_row()] * 101, "nextCursor": None}
            return {"data": [], "nextCursor": f"cursor-{page_number}"}

    client = UnboundedCatalogClient()
    with pytest.raises(CodexCUACatalogDrift, match="limit"):
        _broker(client).call("list_apps", {}, timeout=2.0)
    assert client.requests[-1][0] == "thread/delete"


def test_catalog_pages_share_one_overall_deadline() -> None:
    from agent.transports.codex_cua_broker import CodexCUACatalogDrift

    class SlowCatalogClient(FakeClient):
        def request(self, method, params=None, timeout=30.0):
            if method != "mcpServerStatus/list":
                return super().request(method, params, timeout)
            self.requests.append((method, params or {}))
            time.sleep(0.03)
            return {"data": [], "nextCursor": f"cursor-{len(self.requests)}"}

    client = SlowCatalogClient()
    with pytest.raises(CodexCUACatalogDrift, match="deadline"):
        _broker(client).call("list_apps", {}, timeout=0.05)
    page_timeouts = [
        params for method, params in client.requests if method == "mcpServerStatus/list"
    ]
    assert len(page_timeouts) <= 2


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("slow"),
        RuntimeError("closed"),
        CodexAppServerTransportError("write ambiguous"),
        __import__("asyncio").CancelledError(),
    ],
)
def test_tool_failure_is_never_replayed_and_deletes_thread(failure) -> None:
    from agent.transports.codex_cua_broker import CodexCUAResultAmbiguous

    client = FakeClient(failure=failure)

    expected = (
        type(failure)
        if isinstance(failure, __import__("asyncio").CancelledError)
        else CodexCUAResultAmbiguous
    )
    with pytest.raises(expected):
        _broker(client).call("click", {"app": "Finder"}, timeout=2.0)

    assert [method for method, _params in client.requests].count(
        "mcpServer/tool/call"
    ) == 1
    assert client.requests[-1][0] == "thread/delete"
    assert client.closed is True


def test_mcp_error_result_is_returned_and_thread_is_deleted() -> None:
    client = FakeClient(tool_result={
        "content": [{"type": "text", "text": "rejected"}],
        "isError": True,
    })

    result = _broker(client).call("list_apps", {}, timeout=2.0)

    assert result["isError"] is True
    assert client.requests[-1][0] == "thread/delete"
    assert client.closed is True


@pytest.mark.parametrize(
    "tool_result",
    [
        {"structuredContent": {"missing": "content"}},
        {"content": "not-a-list", "isError": False},
        {"content": [], "isError": "false"},
        {"content": [{"type": "future/hostile", "payload": "x"}], "isError": False},
        {"content": [{"type": "text"}], "isError": False},
        {"content": [{"type": "image", "data": "x", "mimeType": 3}], "isError": False},
        {
            "content": [{"type": "text", "text": "bad number"}],
            "isError": False,
            "structuredContent": {"value": float("nan")},
        },
    ],
)
def test_malformed_post_dispatch_result_is_ambiguous_and_deletes_thread(tool_result) -> None:
    from agent.transports.codex_cua_broker import CodexCUAResultAmbiguous

    client = FakeClient(tool_result=tool_result)
    with pytest.raises(CodexCUAResultAmbiguous, match="do not retry"):
        _broker(client).call("list_apps", {}, timeout=2.0)
    assert client.requests[-1][0] == "thread/delete"
    assert client.closed is True


def test_direct_sky_auth_error_is_typed_and_never_falls_back() -> None:
    from agent.transports.codex_app_server import CodexAppServerError
    from agent.transports.codex_cua_broker import CodexCUACallRejected

    client = FakeClient(failure=CodexAppServerError(
        code=-10000, message="authentication failed"
    ))

    with pytest.raises(CodexCUACallRejected):
        _broker(client).call("list_apps", {}, timeout=2.0)
    assert [method for method, _params in client.requests].count(
        "mcpServer/tool/call"
    ) == 1
    assert client.requests[-1][0] == "thread/delete"


def test_generic_rpc_error_after_dispatch_is_ambiguous_and_never_retried() -> None:
    from agent.transports.codex_app_server import CodexAppServerError
    from agent.transports.codex_cua_broker import CodexCUAResultAmbiguous

    client = FakeClient(failure=CodexAppServerError(
        code=-32001, message="internal failure", data={"retryable": True}
    ))

    with pytest.raises(CodexCUAResultAmbiguous, match="do not retry"):
        _broker(client).call("click", {"app": "Finder"}, timeout=2.0)
    assert [method for method, _params in client.requests].count(
        "mcpServer/tool/call"
    ) == 1
    assert client.requests[-1][0] == "thread/delete"


def test_cleanup_failure_cannot_hide_a_known_completed_result() -> None:
    client = FakeClient(delete_failure=RuntimeError("delete failed"))

    result = _broker(client).call("list_apps", {}, timeout=2.0)

    assert result["structuredContent"] == {"windows": []}
    assert client.closed is True


def test_connection_close_failure_cannot_hide_a_known_completed_result() -> None:
    client = FakeClient(close_failure=RuntimeError("close failed"))

    result = _broker(client).call("list_apps", {}, timeout=2.0)

    assert result["structuredContent"] == {"windows": []}
    assert client.requests[-1][0] == "thread/delete"
    assert client.closed is True


def test_daemon_or_initialize_version_drift_fails_before_thread() -> None:
    from agent.transports.codex_cua_broker import CodexCUAVersionMismatch

    client = FakeClient()
    with pytest.raises(CodexCUAVersionMismatch):
        _broker(client, daemon=FakeDaemon(version="0.150.0")).call(
            "list_apps", {}, timeout=2.0
        )
    assert client.requests == []
    # Version drift is rejected before a connection is created.
    assert client.closed is False

    class DriftedClient(FakeClient):
        def initialize(self, **kwargs):
            return {
                "userAgent": "codex-cli/0.150.0",
                "codexHome": "/tmp/codex-home",
                "platformFamily": "unix",
                "platformOs": "macos",
            }

    drifted = DriftedClient()
    with pytest.raises(CodexCUAVersionMismatch):
        _broker(drifted).call("list_apps", {}, timeout=2.0)
    assert drifted.requests == []
    assert drifted.closed is True


def test_malformed_initialize_thread_and_catalog_envelopes_fail_closed() -> None:
    from agent.transports.codex_cua_broker import (
        CodexCUACatalogDrift,
        CodexCUAProtocolError,
    )

    class BadInitialize(FakeClient):
        def initialize(self, **kwargs):
            return []

    class BadThread(FakeClient):
        def request(self, method, params=None, timeout=30.0):
            if method == "thread/start":
                self.requests.append((method, params or {}))
                return []
            return super().request(method, params, timeout)

    class BadCatalog(FakeClient):
        def request(self, method, params=None, timeout=30.0):
            if method == "mcpServerStatus/list":
                self.requests.append((method, params or {}))
                return []
            return super().request(method, params, timeout)

    for client, error_type in (
        (BadInitialize(), CodexCUAProtocolError),
        (BadThread(), CodexCUAProtocolError),
        (BadCatalog(), CodexCUACatalogDrift),
    ):
        with pytest.raises(error_type):
            _broker(client).call("list_apps", {}, timeout=2.0)
        assert client.closed is True


def test_app_server_and_sdk_results_share_one_renderer() -> None:
    from types import SimpleNamespace
    from tools import mcp_tool

    raw = {
        "content": [{"type": "text", "text": "hello"}],
        "isError": False,
        "structuredContent": {
            "hits": 3,
            "_meta": {"literalStructuredKey": True},
            "nested": [{"_meta": "keep-verbatim"}],
        },
        "_meta": {
            "com.example/status": {"_meta": "keep-verbatim"},
            "modelcontextprotocol.io/private": "drop",
        },
    }
    sdk = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        is_error=False,
        structured_content={
            "hits": 3,
            "_meta": {"literalStructuredKey": True},
            "nested": [{"_meta": "keep-verbatim"}],
        },
        meta={
            "com.example/status": {"_meta": "keep-verbatim"},
            "modelcontextprotocol.io/private": "drop",
        },
    )

    broker_rendered = mcp_tool._render_mcp_tool_result(
        mcp_tool._mcp_protocol_object(raw), "ordinary", "fixture"
    )
    sdk_rendered = mcp_tool._render_mcp_tool_result(
        sdk, "ordinary", "fixture"
    )

    assert broker_rendered == sdk_rendered
    payload = __import__("json").loads(broker_rendered[0])
    assert payload["structuredContent"] == {
        "hits": 3,
        "_meta": {"literalStructuredKey": True},
        "nested": [{"_meta": "keep-verbatim"}],
    }
    assert payload["_meta"] == {
        "com.example/status": {"_meta": "keep-verbatim"}
    }


def test_full_broker_sequence_over_real_local_uds_rejects_server_requests() -> None:
    from agent.transports.codex_app_server import (
        CodexAppServerClient,
        UnixWebSocketCodexAppServerConnection,
    )
    from agent.transports.codex_cua_broker import CodexCUABroker, VerifiedCodex

    socket_path = Path("/tmp") / f"hermes-cua-{uuid.uuid4().hex[:12]}.sock"
    methods = []
    rejection = []

    def handle(websocket) -> None:
        for raw in websocket:
            message = json.loads(raw)
            method = message.get("method")
            methods.append(method or "response")
            if method == "initialize":
                websocket.send(json.dumps({
                    "id": message["id"],
                    "result": {
                        "userAgent": "codex-cli/0.149.0-alpha.4.3",
                        "codexHome": "/tmp/codex-home",
                        "platformFamily": "unix",
                        "platformOs": "macos",
                    },
                }))
            elif method == "initialized":
                continue
            elif method == "thread/start":
                websocket.send(json.dumps({
                    "id": message["id"],
                    "result": {
                        "thread": {"id": "thread-uds", "ephemeral": True, "path": None}
                    },
                }))
            elif method == "mcpServerStatus/list":
                websocket.send(json.dumps({
                    "id": 701,
                    "method": "mcpServer/elicitation/request",
                    "params": {"threadId": "thread-uds"},
                }))
                websocket.send(json.dumps({
                    "id": message["id"],
                    "result": {"data": [_catalog_row()], "nextCursor": None},
                }))
            elif method == "mcpServer/tool/call":
                websocket.send(json.dumps({
                    "id": message["id"],
                    "result": {
                        "content": [{"type": "text", "text": "ok"}],
                        "isError": False,
                        "structuredContent": {"windows": []},
                    },
                }))
            elif method == "thread/delete":
                websocket.send(json.dumps({"id": message["id"], "result": {}}))
            elif "error" in message and message.get("id") == 701:
                rejection.append(message)

    server = unix_serve(handle, path=str(socket_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        verified = VerifiedCodex(
            path="/Applications/ChatGPT.app/Contents/Resources/codex",
            version="0.149.0-alpha.4.3",
            path_snapshot=(),
        )
        class UDSDaemon(FakeDaemon):
            def ensure_ready(self, verified, *, deadline=None):
                assert deadline is None or deadline > time.monotonic()
                self.calls.append(verified)
                return type("Ready", (), {
                    "socket_path": str(socket_path),
                    "version": self.version,
                })()

        broker = CodexCUABroker(
            binary_resolver=lambda **_kwargs: verified,
            daemon_controller=UDSDaemon(),
            client_factory=lambda endpoint, open_timeout: CodexAppServerClient(
                connection=UnixWebSocketCodexAppServerConnection(
                    endpoint, open_timeout=open_timeout
                ),
                reject_server_requests=True,
                event_queue_limit=4,
            ),
            cwd="/tmp",
        )

        result = broker.call("list_apps", {}, timeout=2.0)

        assert result["structuredContent"] == {"windows": []}
        assert [method for method in methods if method != "response"] == [
            "initialize",
            "initialized",
            "thread/start",
            "mcpServerStatus/list",
            "mcpServer/tool/call",
            "thread/delete",
        ]
        assert rejection == [{
            "id": 701,
            "error": {
                "code": -32601,
                "message": "server-initiated requests are disabled",
            },
        }]
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        socket_path.unlink(missing_ok=True)


def test_authenticated_uds_rejects_unexpected_unsigned_peer_before_handshake() -> None:
    from agent.transports.codex_app_server import UnixWebSocketCodexAppServerConnection
    from agent.transports.codex_cua_broker import (
        CodexCUABinaryUntrusted,
        VerifiedCodex,
        _validate_codex_peer,
    )

    socket_path = Path("/tmp") / f"hermes-cua-peer-{uuid.uuid4().hex[:10]}.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    accepted = threading.Event()

    def accept_once() -> None:
        connection, _ = listener.accept()
        accepted.set()
        try:
            connection.recv(1)
        finally:
            connection.close()

    thread = threading.Thread(target=accept_once, daemon=True)
    thread.start()
    verified = VerifiedCodex(
        path="/Applications/ChatGPT.app/Contents/Resources/codex",
        version="0.149.0-alpha.4.3",
        path_snapshot=(),
    )
    try:
        with pytest.raises(CodexCUABinaryUntrusted, match="peer executable"):
            UnixWebSocketCodexAppServerConnection(
                str(socket_path),
                peer_validator=lambda peer: _validate_codex_peer(peer, verified),
            )
        assert accepted.wait(1.0)
    finally:
        listener.close()
        thread.join(timeout=2.0)
        socket_path.unlink(missing_ok=True)


def test_broker_deadline_bounds_stalled_websocket_upgrade() -> None:
    from agent.transports.codex_app_server import (
        CodexAppServerClient,
        UnixWebSocketCodexAppServerConnection,
    )
    from agent.transports.codex_cua_broker import CodexCUABroker, VerifiedCodex

    socket_path = Path("/tmp") / f"hermes-cua-stall-{uuid.uuid4().hex[:10]}.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen(1)
    release = threading.Event()

    def accept_and_stall() -> None:
        connection, _ = listener.accept()
        try:
            release.wait(1.0)
        finally:
            connection.close()

    thread = threading.Thread(target=accept_and_stall, daemon=True)
    thread.start()
    verified = VerifiedCodex(
        path="/Applications/ChatGPT.app/Contents/Resources/codex",
        version="0.149.0-alpha.4.3",
        path_snapshot=(),
    )

    class StallDaemon(FakeDaemon):
        def ensure_ready(self, verified, *, deadline=None):
            return type("Ready", (), {
                "socket_path": str(socket_path),
                "version": self.version,
            })()

    broker = CodexCUABroker(
        binary_resolver=lambda **_kwargs: verified,
        daemon_controller=StallDaemon(),
        client_factory=lambda endpoint, open_timeout: CodexAppServerClient(
            connection=UnixWebSocketCodexAppServerConnection(
                endpoint, open_timeout=open_timeout
            )
        ),
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            broker.call("list_apps", {}, timeout=0.1)
        assert time.monotonic() - started < 0.5
    finally:
        release.set()
        listener.close()
        thread.join(timeout=2.0)
        socket_path.unlink(missing_ok=True)


def test_client_returned_after_deadline_is_closed_without_protocol_work() -> None:
    from agent.transports.codex_cua_broker import CodexCUABroker, VerifiedCodex

    client = FakeClient()
    verified = VerifiedCodex(
        path="/Applications/ChatGPT.app/Contents/Resources/codex",
        version="0.149.0-alpha.4.3",
        path_snapshot=(),
    )

    def late_factory(_endpoint, open_timeout):
        time.sleep(open_timeout + 0.01)
        return client

    broker = CodexCUABroker(
        binary_resolver=lambda **_kwargs: verified,
        daemon_controller=FakeDaemon(),
        client_factory=late_factory,
    )
    with pytest.raises(TimeoutError, match="socket connection"):
        broker.call("list_apps", {}, timeout=0.05)
    assert client.closed is True
    assert client.requests == []
