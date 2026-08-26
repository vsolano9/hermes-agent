"""Codex app-server JSON-RPC client.

Speaks the protocol documented in codex-rs/app-server/README.md (codex 0.125+).
Transport is JSON-RPC 2.0 over either newline-delimited stdio or a Unix-domain
WebSocket. The stdio connection preserves the optional model runtime; the UDS
connection lets trusted local clients share a separately managed app-server.

This module is the wire-level speaker only. Higher-level concerns (event
projection into Hermes' display, approval bridging, transcript projection into
AIAgent.messages, plugin migration) live in sibling modules.

Status: optional opt-in runtime gated behind `model.openai_runtime ==
"codex_app_server"`. Hermes' default tool dispatch is unchanged when this
runtime is not selected.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from tools.environments.local import hermes_subprocess_env

# Default minimum codex version we test against. The PR sets this from the
# `codex --version` parsed at install time; bumping is a one-line change here.
MIN_CODEX_VERSION = (0, 125, 0)
_DEFAULT_EVENT_QUEUE_LIMIT = 256


@dataclass
class CodexAppServerError(RuntimeError):
    """Raised on JSON-RPC errors from the app-server."""

    code: int
    message: str
    data: Optional[Any] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"codex app-server error {self.code}: {self.message}"


class CodexAppServerTransportError(RuntimeError):
    """Raised when transport loss makes a pending response unknowable."""


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


@dataclass
class _Pending:
    queue: queue.Queue
    method: str
    sent_at: float = field(default_factory=time.time)


class CodexAppServerConnection(Protocol):
    """Text-frame connection used by :class:`CodexAppServerClient`."""

    def send_text(self, payload: str) -> None: ...

    def recv_text(self) -> Optional[str]: ...

    def close(self, timeout: float = 3.0) -> None: ...

    def is_alive(self) -> bool: ...

    def stderr_tail(self, n: int = 20) -> list[str]: ...


class StdioCodexAppServerConnection:
    """Newline-delimited JSON connection over a spawned app-server process."""

    def __init__(self, process: subprocess.Popen) -> None:
        self.process = process
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def send_text(self, payload: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("codex app-server stdin not available")
        try:
            self.process.stdin.write((payload + "\n").encode("utf-8"))
            self.process.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError(
                f"codex app-server stdin closed unexpectedly: {exc}"
            ) from exc

    def recv_text(self) -> Optional[str]:
        if self.process.stdout is None:
            return None
        line = self.process.stdout.readline()
        if not line:
            return None
        return line.decode("utf-8", "replace").strip()

    def close(self, timeout: float = 3.0) -> None:
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self.process.kill()
                self.process.wait(timeout=1.0)
            except Exception:
                pass

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def stderr_tail(self, n: int = 20) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def record_diagnostic(self, line: str) -> None:
        with self._stderr_lock:
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > 500:
                self._stderr_lines = self._stderr_lines[-500:]

    def _read_stderr(self) -> None:
        if self.process.stderr is None:
            return
        try:
            for line in iter(self.process.stderr.readline, b""):
                if not line:
                    break
                self.record_diagnostic(line.decode("utf-8", "replace").rstrip())
        except Exception:  # pragma: no cover - diagnostic path only
            pass


class UnixWebSocketCodexAppServerConnection:
    """JSON text-frame connection to an app-server Unix WebSocket."""

    def __init__(
        self,
        socket_path: str,
        *,
        open_timeout: float = 10.0,
        close_timeout: float = 3.0,
        max_size: int = 16 * 1024 * 1024,
        peer_validator: Optional[Callable[[socket.socket], None]] = None,
    ) -> None:
        from websockets.sync.client import unix_connect

        raw_socket = None
        try:
            if peer_validator is not None:
                raw_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                raw_socket.settimeout(open_timeout)
                raw_socket.connect(socket_path)
                peer_validator(raw_socket)
                raw_socket.settimeout(None)
            self._websocket = unix_connect(
                None if raw_socket is not None else socket_path,
                sock=raw_socket,
                uri="ws://localhost/rpc",
                proxy=None,
                compression=None,
                open_timeout=open_timeout,
                close_timeout=close_timeout,
                max_size=max_size,
            )
        except BaseException:
            if raw_socket is not None:
                raw_socket.close()
            raise
        self._closed = False

    def send_text(self, payload: str) -> None:
        if self._closed:
            raise RuntimeError("codex app-server WebSocket is closed")
        self._websocket.send(payload)

    def recv_text(self) -> Optional[str]:
        if self._closed:
            return None
        try:
            payload = self._websocket.recv()
        except Exception:
            self._closed = True
            raise
        if payload is None:
            return None
        if not isinstance(payload, str):
            raise RuntimeError("codex app-server sent a binary WebSocket frame")
        return payload

    def close(self, timeout: float = 3.0) -> None:
        del timeout  # close_timeout is fixed when websockets opens the connection.
        if self._closed:
            return
        self._closed = True
        self._websocket.close()

    def is_alive(self) -> bool:
        return not self._closed

    def stderr_tail(self, n: int = 20) -> list[str]:
        del n
        return []

    def record_diagnostic(self, line: str) -> None:
        del line


class CodexAppServerClient:
    """Minimal synchronous JSON-RPC 2.0 client for `codex app-server`.

    Threading model:
      - Spawning thread (caller) drives request/response pairs synchronously.
      - One reader thread parses stdout, dispatches replies to the right
        pending future, and routes notifications + server-initiated requests
        to bounded queues that the caller drains on their own cadence.
      - One reader thread captures stderr for diagnostics; codex emits
        tracing logs there at RUST_LOG-controlled levels.

    Intentionally NOT async. AIAgent.run_conversation() is synchronous and
    runs on the main thread; layering asyncio just to drive a stdio child
    creates surprising interrupt semantics. We use blocking queues with
    timeouts and rely on `turn/interrupt` for cancellation.
    """

    def __init__(
        self,
        codex_bin: str = "codex",
        codex_home: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        connection: Optional[CodexAppServerConnection] = None,
        reject_server_requests: bool = False,
        event_queue_limit: int = _DEFAULT_EVENT_QUEUE_LIMIT,
        discard_notifications: bool = False,
    ) -> None:
        self._codex_bin = codex_bin
        # codex app-server is a model-driving CLI executor: it runs a
        # model-chosen agentic loop that executes shell commands, so it
        # legitimately needs LLM provider credentials (inherit_credentials=True)
        # to authenticate against the model endpoint. But the previous
        # `os.environ.copy()` also handed it every Tier-1 Hermes secret — gateway
        # bot tokens, GitHub auth, Modal/Daytona infra tokens, the dashboard
        # session token, AUXILIARY_* side-LLM keys, GATEWAY_RELAY_* auth — none
        # of which a coding subprocess has any use for. Route through the
        # centralized helper so Tier-1 + dynamic-internal secrets are always
        # stripped while provider creds still flow, matching copilot_acp_client
        # (#29157 sibling spawn-site gap).
        app_server_args = list(extra_args or [])
        # Kanban workers must be able to write their handoff/status back to
        # the board DB, which lives outside the per-task workspace. Keep the
        # Codex sandbox on, but add the Kanban root as the only extra writable
        # root. Without this, codex-runtime workers finish their actual work
        # but crash/block when kanban_complete/kanban_block writes SQLite.
        spawn_env = hermes_subprocess_env(inherit_credentials=True)
        if env:
            spawn_env.update(env)
        if codex_home:
            spawn_env["CODEX_HOME"] = codex_home
        if spawn_env.get("HERMES_KANBAN_TASK"):
            kanban_db = spawn_env.get("HERMES_KANBAN_DB")
            kanban_root = (
                os.path.dirname(kanban_db)
                if kanban_db
                else spawn_env.get(
                    "HERMES_KANBAN_ROOT",
                    os.path.join(
                        spawn_env.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
                        "kanban",
                    ),
                )
            )
            app_server_args.extend(
                [
                    "-c",
                    'sandbox_mode="workspace-write"',
                    "-c",
                    f'sandbox_workspace_write.writable_roots=["{kanban_root}"]',
                    "-c",
                    "sandbox_workspace_write.network_access=false",
                ]
            )

        cmd = [codex_bin, "app-server"] + app_server_args
        # Codex emits tracing to stderr; default WARN keeps it quiet for users.
        spawn_env.setdefault("RUST_LOG", "warn")

        # Hide the console the codex child would otherwise flash on Windows
        # (#56747). Hide-only — stdio pipes stay intact for the app-server wire.
        from hermes_cli._subprocess_compat import windows_hide_flags

        if connection is None:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=spawn_env,
                creationflags=windows_hide_flags(),
            )
            connection = StdioCodexAppServerConnection(proc)
        self._connection = connection
        # Compatibility for callers which inspect the optional runtime child.
        self._proc = getattr(connection, "process", None)
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        if not isinstance(event_queue_limit, int) or event_queue_limit <= 0:
            raise ValueError("event_queue_limit must be a positive integer")
        self._notifications: queue.Queue = queue.Queue(maxsize=event_queue_limit)
        self._server_requests: queue.Queue = queue.Queue(maxsize=event_queue_limit)
        self._reject_server_requests = reject_server_requests
        # Model runtimes consume the bounded notification queue. Narrow
        # synchronous protocol clients may opt out only when events are not
        # part of their contract; frame and method validation still runs.
        self._discard_notifications = discard_notifications
        self._send_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._initialized = False

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    # ---------- lifecycle ----------

    def initialize(
        self,
        client_name: str = "hermes",
        client_title: str = "Hermes Agent",
        client_version: str = "0.1",
        capabilities: Optional[dict] = None,
        timeout: float = 10.0,
    ) -> dict:
        """Send `initialize` + `initialized` handshake. Returns the server's
        InitializeResponse (userAgent, codexHome, platformFamily, platformOs)."""
        if self._initialized:
            raise RuntimeError("already initialized")
        params = {
            "clientInfo": {
                "name": client_name,
                "title": client_title,
                "version": client_version,
            },
            "capabilities": capabilities or {},
        }
        result = self.request("initialize", params, timeout=timeout)
        self.notify("initialized")
        self._initialized = True
        return result

    def close(self, timeout: float = 3.0) -> None:
        """Close the underlying transport without leaking its resources."""
        self._terminate_transport(
            CodexAppServerTransportError("codex app-server client closed"),
            timeout=timeout,
            suppress_close_error=False,
        )

    def __enter__(self) -> "CodexAppServerClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- send/receive ----------

    def request(
        self,
        method: str,
        params: Optional[dict] = None,
        timeout: float = 30.0,
    ) -> dict:
        """Send a JSON-RPC request and block on the response. Returns `result`,
        raises CodexAppServerError on `error`."""
        rid = self._take_id()
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = _Pending(queue=q, method=method)
        try:
            self._send({"id": rid, "method": method, "params": params or {}})
        except BaseException:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(
                f"codex app-server method {method!r} timed out after {timeout}s"
            )
        if isinstance(msg, BaseException):
            raise msg
        if "error" in msg:
            err = msg["error"]
            raise CodexAppServerError(
                code=err.get("code", -1),
                message=err.get("message", ""),
                data=err.get("data"),
            )
        return msg.get("result", {})

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        self._send({"method": method, "params": params or {}})

    def respond(self, request_id: Any, result: dict) -> None:
        """Reply to a server-initiated request (e.g. approval prompts)."""
        self._send({"id": request_id, "result": result})

    def respond_error(
        self, request_id: Any, code: int, message: str, data: Optional[Any] = None
    ) -> None:
        """Reply to a server-initiated request with an error."""
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"id": request_id, "error": err})

    def take_notification(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next streaming notification, or return None on timeout.

        timeout=0.0 means non-blocking. Use small positive timeouts inside the
        AIAgent turn loop to interleave reads with interrupt checks."""
        try:
            if timeout <= 0:
                return self._notifications.get_nowait()
            return self._notifications.get(timeout=timeout)
        except queue.Empty:
            return None

    def take_server_request(self, timeout: float = 0.0) -> Optional[dict]:
        """Pop the next server-initiated request (e.g. exec/applyPatch approval)."""
        try:
            if timeout <= 0:
                return self._server_requests.get_nowait()
            return self._server_requests.get(timeout=timeout)
        except queue.Empty:
            return None

    # ---------- diagnostics ----------

    def stderr_tail(self, n: int = 20) -> list[str]:
        """Return last n lines of codex's stderr (for error reports)."""
        return self._connection.stderr_tail(n)

    def is_alive(self) -> bool:
        return self._connection.is_alive()

    # ---------- internals ----------

    def _take_id(self) -> int:
        # JSON-RPC ids only need to be unique per-connection. A simple
        # monotonically increasing int is the common choice and matches what
        # codex's own clients use.
        rid = self._next_id
        self._next_id += 1
        return rid

    def _send(self, obj: dict) -> None:
        if self._closed:
            raise RuntimeError("codex app-server client is closed")
        try:
            payload = json.dumps(obj, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("codex app-server request is not JSON serializable") from exc
        try:
            with self._send_lock:
                self._connection.send_text(payload)
        except Exception as exc:
            # WebSocket ConnectionClosed/OSError and stdio pipe failures can
            # occur after some or all frame bytes were accepted. Classify
            # every transport write failure as ambiguous at the wire boundary.
            raise CodexAppServerTransportError(
                "codex app-server transport failed while sending a request"
            ) from exc

    def _read_stdout(self) -> None:
        failure: Optional[CodexAppServerTransportError] = None
        try:
            while True:
                line = self._connection.recv_text()
                if line is None:
                    if not self._closed:
                        failure = CodexAppServerTransportError(
                            "codex app-server connection closed before response"
                        )
                    break
                if not line:
                    continue
                try:
                    msg = json.loads(line, parse_constant=_reject_nonfinite_json)
                except (json.JSONDecodeError, ValueError):
                    recorder = getattr(self._connection, "record_diagnostic", None)
                    if recorder is not None:
                        recorder(f"<non-json on transport> {line[:200]!r}")
                    failure = CodexAppServerTransportError(
                        "codex app-server sent an invalid JSON frame"
                    )
                    break
                if not isinstance(msg, dict):
                    failure = CodexAppServerTransportError(
                        "codex app-server sent a non-object JSON-RPC frame"
                    )
                    break
                self._dispatch(msg)
        except Exception as exc:
            recorder = getattr(self._connection, "record_diagnostic", None)
            if recorder is not None:
                recorder(f"<transport reader error> {exc}")
            failure = CodexAppServerTransportError(
                "codex app-server transport failed before a response"
            )
        finally:
            if failure is not None:
                self._terminate_transport(
                    failure, timeout=3.0, suppress_close_error=True
                )

    def _terminate_transport(
        self,
        failure: BaseException,
        *,
        timeout: float,
        suppress_close_error: bool,
    ) -> None:
        """Atomically make transport loss terminal for all future requests."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._fail_pending(failure)
            try:
                self._connection.close(timeout=timeout)
            except Exception as exc:
                if not suppress_close_error:
                    raise
                recorder = getattr(self._connection, "record_diagnostic", None)
                if recorder is not None:
                    recorder(f"<transport close error> {exc}")

    def _fail_pending(self, failure: BaseException) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            try:
                item.queue.put_nowait(failure)
            except queue.Full:  # pragma: no cover - defensive
                pass

    def _dispatch(self, msg: dict) -> None:
        # Reply (has id + result/error, no method)
        if "id" in msg and ("result" in msg or "error" in msg):
            if "result" in msg and "error" in msg:
                raise CodexAppServerTransportError(
                    "codex app-server reply contains both result and error"
                )
            if "error" in msg:
                error = msg["error"]
                if (
                    not isinstance(error, dict)
                    or not isinstance(error.get("code"), int)
                    or not isinstance(error.get("message"), str)
                ):
                    raise CodexAppServerTransportError(
                        "codex app-server sent a malformed JSON-RPC error"
                    )
            with self._pending_lock:
                try:
                    pending = self._pending.pop(msg["id"], None)
                except TypeError as exc:
                    raise CodexAppServerTransportError(
                        "codex app-server sent an invalid JSON-RPC id"
                    ) from exc
            if pending is not None:
                try:
                    pending.queue.put_nowait(msg)
                except queue.Full:  # pragma: no cover - defensive
                    pass
            return
        # Server-initiated request (has id + method)
        if "id" in msg and "method" in msg:
            if not isinstance(msg["method"], str) or not msg["method"]:
                raise CodexAppServerTransportError(
                    "codex app-server sent an invalid JSON-RPC method"
                )
            if self._reject_server_requests:
                self.respond_error(
                    msg["id"], -32601, "server-initiated requests are disabled"
                )
                return
            try:
                self._server_requests.put_nowait(msg)
            except queue.Full as exc:
                raise CodexAppServerTransportError(
                    "codex app-server server-request queue overflowed"
                ) from exc
            return
        # Notification (no id)
        if "method" in msg:
            if not isinstance(msg["method"], str) or not msg["method"]:
                raise CodexAppServerTransportError(
                    "codex app-server sent an invalid JSON-RPC method"
                )
            if "result" in msg or "error" in msg:
                raise CodexAppServerTransportError(
                    "codex app-server sent a malformed JSON-RPC notification"
                )
            if "params" in msg and not isinstance(msg["params"], (dict, list)):
                raise CodexAppServerTransportError(
                    "codex app-server sent invalid JSON-RPC notification params"
                )
            if self._discard_notifications:
                return
            try:
                self._notifications.put_nowait(msg)
            except queue.Full as exc:
                raise CodexAppServerTransportError(
                    "codex app-server notification queue overflowed"
                ) from exc
            return
        raise CodexAppServerTransportError(
            "codex app-server sent an unrecognized JSON-RPC envelope"
        )

def parse_codex_version(output: str) -> Optional[tuple[int, int, int]]:
    """Parse `codex --version` output. Returns (major, minor, patch) or None."""
    # Output format: "codex-cli 0.130.0" possibly followed by metadata.
    import re

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def check_codex_binary(
    codex_bin: str = "codex", min_version: tuple[int, int, int] = MIN_CODEX_VERSION
) -> tuple[bool, str]:
    """Verify codex CLI is installed and meets minimum version.

    Returns (ok, message). Used by setup wizard and runtime startup."""
    try:
        proc = subprocess.run(
            [codex_bin, "--version"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, (
            f"codex CLI not found at {codex_bin!r}. Install with: "
            f"npm i -g @openai/codex"
        )
    except subprocess.TimeoutExpired:
        return False, "codex --version timed out"
    if proc.returncode != 0:
        return False, f"codex --version exited {proc.returncode}: {proc.stderr.strip()}"
    version = parse_codex_version(proc.stdout)
    if version is None:
        return False, f"could not parse codex version from: {proc.stdout!r}"
    if version < min_version:
        return False, (
            f"codex {'.'.join(map(str, version))} is older than required "
            f"{'.'.join(map(str, min_version))}. Run: npm i -g @openai/codex"
        )
    return True, ".".join(map(str, version))
