"""Behavior tests for Codex app-server over a Unix WebSocket."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import uuid

import pytest
from websockets.sync.server import unix_serve

from agent.transports import codex_app_server as app_server
from agent.transports.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    CodexAppServerTransportError,
)


def test_uds_connection_preserves_the_json_rpc_client_contract(tmp_path) -> None:
    """A UDS connection supports the same app-server contract as stdio."""

    # macOS caps AF_UNIX paths at roughly 104 bytes; pytest's hermetic temp
    # root is deliberately much longer than that.
    socket_path = Path("/tmp") / f"hermes-cas-{uuid.uuid4().hex[:12]}.sock"
    initialized = threading.Event()
    request_paths = []
    peer_pids = []

    def handle(websocket) -> None:
        request_paths.append(websocket.request.path)
        for raw in websocket:
            request = json.loads(raw)
            method = request.get("method")
            if method == "initialize":
                websocket.send(json.dumps({
                    "id": request["id"],
                    "result": {
                        "userAgent": "codex-cli/0.149.0-alpha.4.3",
                        "codexHome": "/tmp/codex-home",
                        "platformFamily": "unix",
                        "platformOs": "macos",
                    },
                }))
            elif method == "initialized":
                initialized.set()
                websocket.send(json.dumps({
                    "method": "thread/started",
                    "params": {"thread": {"id": "thread-1"}},
                }))
                websocket.send(json.dumps({
                    "id": 71,
                    "method": "mcpServer/elicitation/request",
                    "params": {"threadId": "thread-1"},
                }))
            elif method == "echo":
                websocket.send(json.dumps({
                    "id": request["id"],
                    "result": {"value": request["params"]["value"]},
                }))
            elif method == "explode":
                websocket.send(json.dumps({
                    "id": request["id"],
                    "error": {
                        "code": -32001,
                        "message": "fixture exploded",
                        "data": {"retryable": False},
                    },
                }))

    server = unix_serve(handle, path=str(socket_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = app_server.UnixWebSocketCodexAppServerConnection(
            str(socket_path),
            peer_validator=lambda peer: peer_pids.append(peer.getsockopt(0, 0x002)),
        )
        client = CodexAppServerClient(connection=connection)

        initialized_result = client.initialize(timeout=2.0)
        assert request_paths == ["/rpc"]
        assert peer_pids == [__import__("os").getpid()]
        assert initialized_result["platformOs"] == "macos"
        assert initialized.wait(1.0)
        assert client.request("echo", {"value": 42}, timeout=2.0) == {"value": 42}

        note = client.take_notification(timeout=1.0)
        assert note == {
            "method": "thread/started",
            "params": {"thread": {"id": "thread-1"}},
        }
        request = client.take_server_request(timeout=1.0)
        assert request == {
            "id": 71,
            "method": "mcpServer/elicitation/request",
            "params": {"threadId": "thread-1"},
        }

        with pytest.raises(CodexAppServerError) as exc_info:
            client.request("explode", timeout=2.0)
        assert exc_info.value.code == -32001
        assert exc_info.value.data == {"retryable": False}

        client.close()
        assert not connection.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            client.request("echo", {"value": 7}, timeout=0.1)
    finally:
        server.shutdown()
        thread.join(timeout=2.0)
        socket_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "payload,max_size",
    [
        (b"binary", 1024),
        ("not-json", 1024),
        ("[]", 1024),
        ("true", 1024),
        ('{"id":1,"result":{"value":NaN}}', 1024),
        ('{"id":1,"error":"not-an-error-object"}', 1024),
        ("x" * 2048, 128),
    ],
)
def test_invalid_uds_frame_fails_pending_request_without_waiting_for_timeout(
    payload, max_size
) -> None:
    socket_path = Path("/tmp") / f"hermes-cas-{uuid.uuid4().hex[:12]}.sock"

    def handle(websocket) -> None:
        websocket.recv()
        websocket.send(payload)
        websocket.close()

    server = unix_serve(handle, path=str(socket_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = None
    try:
        connection = app_server.UnixWebSocketCodexAppServerConnection(
            str(socket_path), max_size=max_size
        )
        client = CodexAppServerClient(connection=connection)

        with pytest.raises(CodexAppServerTransportError):
            client.request("fixture/fail", timeout=5.0)
        with pytest.raises(RuntimeError, match="closed"):
            client.request("fixture/again", timeout=5.0)
        assert connection.is_alive() is False
    finally:
        if client is not None:
            client.close()
        server.shutdown()
        thread.join(timeout=2.0)
        socket_path.unlink(missing_ok=True)


def test_write_side_transport_failure_is_typed_without_waiting() -> None:
    stopped = threading.Event()

    class FailingWriteConnection:
        def send_text(self, payload):
            raise OSError("write state is unknowable")

        def recv_text(self):
            stopped.wait(2.0)
            return None

        def close(self, timeout=3.0):
            stopped.set()

        def is_alive(self):
            return not stopped.is_set()

        def stderr_tail(self, n=20):
            return []

    client = CodexAppServerClient(connection=FailingWriteConnection())
    try:
        with pytest.raises(CodexAppServerTransportError, match="sending"):
            client.request("fixture/write", timeout=5.0)
    finally:
        client.close()


def test_nonfinite_outbound_json_is_rejected_before_transport_write() -> None:
    stopped = threading.Event()

    class RecordingConnection:
        writes = []

        def send_text(self, payload):
            self.writes.append(payload)

        def recv_text(self):
            stopped.wait(1.0)
            return None

        def close(self, timeout=3.0):
            del timeout
            stopped.set()

        def is_alive(self):
            return not stopped.is_set()

        def stderr_tail(self, n=20):
            del n
            return []

    connection = RecordingConnection()
    client = CodexAppServerClient(connection=connection)
    try:
        with pytest.raises(ValueError, match="serializable"):
            client.request("fixture/nonfinite", {"value": float("nan")})
        assert connection.writes == []
    finally:
        client.close()


def test_notification_queue_overflow_is_terminal_and_memory_bounded() -> None:
    frames: queue.Queue = queue.Queue()
    closed = threading.Event()

    class FloodConnection:
        def send_text(self, payload):
            del payload

        def recv_text(self):
            return frames.get(timeout=2.0)

        def close(self, timeout=3.0):
            del timeout
            closed.set()

        def is_alive(self):
            return not closed.is_set()

        def stderr_tail(self, n=20):
            del n
            return []

    for index in range(12):
        frames.put(json.dumps({"method": "fixture/noise", "params": {"n": index}}))

    client = CodexAppServerClient(
        connection=FloodConnection(), event_queue_limit=4
    )
    try:
        assert closed.wait(1.0), "a non-consuming client must fail on bounded queue overflow"
        with pytest.raises(RuntimeError, match="closed"):
            client.request("fixture/after-flood", timeout=1.0)
    finally:
        client.close()


def test_default_runtime_queue_preserves_legitimate_burst_order() -> None:
    frames: queue.Queue = queue.Queue()
    closed = threading.Event()

    class BurstConnection:
        def send_text(self, payload):
            del payload

        def recv_text(self):
            return frames.get(timeout=2.0)

        def close(self, timeout=3.0):
            del timeout
            closed.set()

        def is_alive(self):
            return not closed.is_set()

        def stderr_tail(self, n=20):
            del n
            return []

    for index in range(32):
        frames.put(json.dumps({"method": "turn/item/delta", "params": {"n": index}}))

    client = CodexAppServerClient(connection=BurstConnection())
    try:
        received = [client.take_notification(timeout=1.0) for _ in range(32)]
        assert [item["params"]["n"] for item in received] == list(range(32))
        assert client.is_alive()
    finally:
        client.close()


def test_server_request_rejection_policy_responds_without_deadlock() -> None:
    socket_path = Path("/tmp") / f"hermes-cas-{uuid.uuid4().hex[:12]}.sock"
    rejected = queue.Queue(maxsize=1)

    def handle(websocket) -> None:
        initialize = json.loads(websocket.recv())
        websocket.send(json.dumps({
            "id": 99,
            "method": "mcpServer/elicitation/request",
            "params": {"threadId": "thread-1"},
        }))
        rejected.put(json.loads(websocket.recv()))
        websocket.send(json.dumps({
            "id": initialize["id"],
            "result": {
                "userAgent": "codex-cli/0.149.0-alpha.4.3",
                "codexHome": "/tmp/codex-home",
                "platformFamily": "unix",
                "platformOs": "macos",
            },
        }))
        websocket.recv()  # initialized notification

    server = unix_serve(handle, path=str(socket_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = None
    try:
        connection = app_server.UnixWebSocketCodexAppServerConnection(str(socket_path))
        client = CodexAppServerClient(
            connection=connection, reject_server_requests=True
        )
        client.initialize(timeout=2.0)
        response = rejected.get(timeout=1.0)
        assert response == {
            "id": 99,
            "error": {
                "code": -32601,
                "message": "server-initiated requests are disabled",
            },
        }
        assert client.take_server_request() is None
    finally:
        if client is not None:
            client.close()
        server.shutdown()
        thread.join(timeout=2.0)
        socket_path.unlink(missing_ok=True)
