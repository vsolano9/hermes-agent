"""Public, plugin-safe lifecycle API for delegated Hermes subagents.

This module deliberately exposes immutable contracts, not ``AIAgent`` objects.
It is the supported boundary for plugins that need to supervise fresh child
sessions; plugins must obtain it from ``PluginContext.subagent_lifecycle``.
"""

from __future__ import annotations

import contextvars
import dataclasses
import enum
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, Iterator, Mapping, Optional

from agent.interrupt_compat import request_hard_interrupt

PUBLIC_CONTRACT_VERSION = 1
LIFECYCLE_API_CONTRACT_VERSION = 3
_SUPPORTED_LAUNCH_REQUEST_API_CONTRACT_VERSIONS = frozenset({2, 3})
_MAX_GOAL_CHARS = 16_000
_MAX_CONTEXT_CHARS = 32_000
_MAX_METADATA_BYTES = 8_192
_MAX_RESULT_CHARS = 32_000
_MAX_API_CALLS = 1_000_000
_MAX_DURATION_SECONDS = 31_536_000.0
_TERMINAL_RETENTION_SECONDS = 3_600
_AUDIT_UNSET = object()
_MAX_ROUTE_IDENTIFIER_CHARS = 200
_MAX_ROUTE_CANDIDATES = 512
_MAX_PUBLIC_TIMESTAMP_SECONDS = 253_402_300_799.0
_ROUTE_REASONS = frozenset(
    {
        "COMPLETE",
        "CATALOG_UNAVAILABLE",
        "CATALOG_INCOMPLETE",
        "ELIGIBLE",
        "MUTATION_CHANNEL_UNAVAILABLE",
        "ROUTE_UNAVAILABLE",
    }
)
_ROUTE_MUTATION_CHANNELS = frozenset(
    {
        "ACP_FILESYSTEM",
        "EXTERNAL_PROCESS",
        "UNKNOWN_TRANSPORT",
        "HERMES_MODEL_TOOLS",
    }
)
_ROUTE_TRANSPORTS = frozenset(
    {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
        "unknown",
        "unavailable",
    }
)


class _FrozenMapping(Mapping[str, Any]):
    """Recursively immutable mapping used in durable result receipts."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_items", tuple(values.items()))

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("immutable mapping")

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_FrozenMapping":
        return self


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_deep_freeze(item) for item in value), key=repr))
    if isinstance(value, bytearray):
        return bytes(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _FrozenMapping(
            {
                field.name: _deep_freeze(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
        )
    if value is None or isinstance(value, (str, bytes, int, float, bool, enum.Enum)):
        return value
    return str(value)[:_MAX_RESULT_CHARS]


def _canonical_hash_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_hash_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name != "result_hash"
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_hash_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_hash_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical_hash_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, bytes):
        return value.hex()
    return value


def _freeze_structured_payload(value: Mapping[str, Any]) -> _FrozenMapping:
    frozen = _deep_freeze(value)
    canonical = json.dumps(
        _canonical_hash_value(frozen),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    if len(canonical) <= _MAX_METADATA_BYTES:
        return frozen
    return _FrozenMapping(
        {
            "truncated": True,
            "content_hash": hashlib.sha256(canonical).hexdigest(),
            "original_bytes": len(canonical),
        }
    )


def _bounded_api_calls(value: Any) -> int:
    if type(value) is not int or value < 0:
        return 0
    return min(value, _MAX_API_CALLS)


def _bounded_duration_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return 0.0
    return min(numeric, _MAX_DURATION_SECONDS)


class SubagentLifecycleError(ValueError):
    """A request cannot be safely accepted by the public lifecycle API."""


def _public_route_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise SubagentLifecycleError(f"{field} must be a string identifier.")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_ROUTE_IDENTIFIER_CHARS
        or not normalized[0].isalnum()
        or not normalized[0].isascii()
        or "://" in normalized
        or any(
            not character.isascii()
            or not (character.isalnum() or character in {"-", "_", ".", ":", "/", "@", "+"})
            for character in normalized
        )
    ):
        raise SubagentLifecycleError(
            f"{field} must be a bounded public identifier."
        )
    return normalized


def _public_route_receipt_id(value: Any, *, prefix: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != len(prefix) + 32
        or not value.startswith(prefix)
        or any(character not in "0123456789abcdef" for character in value[len(prefix) :])
    ):
        raise SubagentLifecycleError("Malformed public route receipt identity.")
    return value


def _public_route_assessed_at(value: Any) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0 < value <= _MAX_PUBLIC_TIMESTAMP_SECONDS:
        raise SubagentLifecycleError("Malformed public route assessment timestamp.")
    return value


def _new_public_route_receipt_metadata(*, prefix: str) -> tuple[float, str]:
    assessed_at = float(time.time())
    if not math.isfinite(assessed_at) or assessed_at <= 0:
        assessed_at = 0.000001
    else:
        assessed_at = min(assessed_at, _MAX_PUBLIC_TIMESTAMP_SECONDS)
    return assessed_at, f"{prefix}{secrets.token_hex(16)}"


@dataclasses.dataclass(frozen=True)
class SubagentRouteIdentity:
    provider: str
    model: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider", _public_route_identifier(self.provider, field="provider")
        )
        object.__setattr__(
            self, "model", _public_route_identifier(self.model, field="model")
        )


@dataclasses.dataclass(frozen=True)
class SubagentRouteAssessment:
    api_contract_version: int
    route: SubagentRouteIdentity
    eligible: bool
    reason: str
    transport: str
    authenticated: bool
    agent_capable: bool
    exact_empty_model_tools: bool
    mutation_evidence_complete: bool
    independent_mutation_channels: frozenset[str]
    hermes_model_tool_count: int
    assessed_at: float
    assessment_id: str

    def __post_init__(self) -> None:
        if self.api_contract_version != LIFECYCLE_API_CONTRACT_VERSION:
            raise SubagentLifecycleError("Unsupported lifecycle API contract version.")
        if not isinstance(self.route, SubagentRouteIdentity):
            raise SubagentLifecycleError("Malformed subagent route identity.")
        if (
            type(self.eligible) is not bool
            or self.reason not in _ROUTE_REASONS
            or self.transport not in _ROUTE_TRANSPORTS
            or any(
                type(flag) is not bool
                for flag in (
                    self.authenticated,
                    self.agent_capable,
                    self.exact_empty_model_tools,
                    self.mutation_evidence_complete,
                )
            )
        ):
            raise SubagentLifecycleError("Malformed subagent route assessment.")
        channels = frozenset(self.independent_mutation_channels)
        if not channels <= _ROUTE_MUTATION_CHANNELS:
            raise SubagentLifecycleError("Malformed subagent route assessment.")
        if (
            type(self.hermes_model_tool_count) is not int
            or not 0 <= self.hermes_model_tool_count <= _MAX_ROUTE_CANDIDATES
        ):
            raise SubagentLifecycleError("Malformed subagent route assessment.")
        _public_route_assessed_at(self.assessed_at)
        _public_route_receipt_id(self.assessment_id, prefix="asm_")
        unknown_transport = self.transport == "unknown"
        unknown_transport_channel = "UNKNOWN_TRANSPORT" in channels
        model_tools_channel = "HERMES_MODEL_TOOLS" in channels
        if unknown_transport != unknown_transport_channel:
            raise SubagentLifecycleError("Malformed subagent route assessment.")
        if self.mutation_evidence_complete:
            if (
                self.exact_empty_model_tools
                != (self.hermes_model_tool_count == 0)
                or model_tools_channel != (self.hermes_model_tool_count > 0)
            ):
                raise SubagentLifecycleError("Malformed subagent route assessment.")
        elif (
            self.exact_empty_model_tools
            or self.hermes_model_tool_count != 0
            or model_tools_channel
        ):
            raise SubagentLifecycleError("Malformed subagent route assessment.")
        if self.eligible:
            if (
                self.reason != "ELIGIBLE"
                or self.transport in {"unknown", "unavailable"}
                or not self.authenticated
                or not self.agent_capable
                or not self.exact_empty_model_tools
                or not self.mutation_evidence_complete
                or channels
                or self.hermes_model_tool_count != 0
            ):
                raise SubagentLifecycleError("Malformed subagent route assessment.")
        elif self.reason == "ROUTE_UNAVAILABLE":
            if (
                self.transport != "unavailable"
                or self.authenticated
                or self.agent_capable
                or self.exact_empty_model_tools
                or self.mutation_evidence_complete
                or channels
                or self.hermes_model_tool_count != 0
            ):
                raise SubagentLifecycleError("Malformed subagent route assessment.")
        elif self.reason == "MUTATION_CHANNEL_UNAVAILABLE":
            if (
                self.transport == "unavailable"
                or not self.authenticated
                or not self.agent_capable
                or not (channels or not self.mutation_evidence_complete)
            ):
                raise SubagentLifecycleError("Malformed subagent route assessment.")
        else:
            raise SubagentLifecycleError("Malformed subagent route assessment.")
        object.__setattr__(self, "independent_mutation_channels", channels)


@dataclasses.dataclass(frozen=True)
class SubagentRouteCatalog:
    api_contract_version: int
    complete: bool
    routes: tuple[SubagentRouteIdentity, ...]
    candidate_count: int
    reason: str
    assessed_at: float
    snapshot_id: str

    def __post_init__(self) -> None:
        routes = tuple(self.routes)
        if self.api_contract_version != LIFECYCLE_API_CONTRACT_VERSION:
            raise SubagentLifecycleError("Unsupported lifecycle API contract version.")
        if type(self.complete) is not bool or self.reason not in _ROUTE_REASONS:
            raise SubagentLifecycleError("Malformed subagent route catalog.")
        _public_route_assessed_at(self.assessed_at)
        _public_route_receipt_id(self.snapshot_id, prefix="snap_")
        if (
            any(not isinstance(route, SubagentRouteIdentity) for route in routes)
            or len(routes) > _MAX_ROUTE_CANDIDATES
            or type(self.candidate_count) is not int
            or not 0 <= self.candidate_count <= _MAX_ROUTE_CANDIDATES
        ):
            raise SubagentLifecycleError("Malformed subagent route catalog.")
        if self.complete:
            if self.reason != "COMPLETE" or self.candidate_count != len(routes):
                raise SubagentLifecycleError("Malformed subagent route catalog.")
        elif routes or self.reason not in {"CATALOG_UNAVAILABLE", "CATALOG_INCOMPLETE"}:
            raise SubagentLifecycleError("Malformed subagent route catalog.")
        object.__setattr__(self, "routes", routes)


def _new_route_catalog(
    *,
    complete: bool,
    routes: tuple[SubagentRouteIdentity, ...],
    candidate_count: int,
    reason: str,
) -> SubagentRouteCatalog:
    assessed_at, snapshot_id = _new_public_route_receipt_metadata(prefix="snap_")
    return SubagentRouteCatalog(
        api_contract_version=LIFECYCLE_API_CONTRACT_VERSION,
        complete=complete,
        routes=routes,
        candidate_count=candidate_count,
        reason=reason,
        assessed_at=assessed_at,
        snapshot_id=snapshot_id,
    )


def _new_route_assessment(
    *,
    route: SubagentRouteIdentity,
    eligible: bool,
    reason: str,
    transport: str,
    authenticated: bool,
    agent_capable: bool,
    exact_empty_model_tools: bool,
    mutation_evidence_complete: bool,
    independent_mutation_channels: frozenset[str],
    hermes_model_tool_count: int,
) -> SubagentRouteAssessment:
    assessed_at, assessment_id = _new_public_route_receipt_metadata(prefix="asm_")
    return SubagentRouteAssessment(
        api_contract_version=LIFECYCLE_API_CONTRACT_VERSION,
        route=route,
        eligible=eligible,
        reason=reason,
        transport=transport,
        authenticated=authenticated,
        agent_capable=agent_capable,
        exact_empty_model_tools=exact_empty_model_tools,
        mutation_evidence_complete=mutation_evidence_complete,
        independent_mutation_channels=independent_mutation_channels,
        hermes_model_tool_count=hermes_model_tool_count,
        assessed_at=assessed_at,
        assessment_id=assessment_id,
    )


class SubagentState(str, enum.Enum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class SubagentControlDisposition(str, enum.Enum):
    QUEUED = "QUEUED"
    MISSED = "MISSED"
    TERMINAL = "TERMINAL"
    UNKNOWN_HANDLE = "UNKNOWN_HANDLE"
    WRONG_AUTHORITY = "WRONG_AUTHORITY"
    UNSUPPORTED = "UNSUPPORTED"


@dataclasses.dataclass(frozen=True)
class SubagentLaunchRequest:
    goal: str
    context: Optional[str] = None
    role: str = "leaf"
    model: Optional[str] = None
    allowed_toolsets: Optional[tuple[str, ...]] = None
    blocked_tools: tuple[str, ...] = ()
    working_directory: Optional[str] = None
    parent_session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    timeout_seconds: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class SubagentLaunchRequestV2:
    api_contract_version: int
    base: SubagentLaunchRequest
    toolset_mode: str = "inherit"
    exact_toolsets: tuple[str, ...] = ()
    provider: Optional[str] = None
    reasoning_effort: Any = None


@dataclasses.dataclass(frozen=True)
class SubagentHandle:
    contract_version: int
    subagent_id: str
    parent_session_id: Optional[str]
    correlation_id: Optional[str]
    created_at: float
    provider: Optional[str]
    model: Optional[str]
    role: str
    depth: int
    capability: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentHandle":
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise SubagentLifecycleError("Malformed subagent handle.") from exc


@dataclasses.dataclass(frozen=True)
class SubagentAuditMetadata:
    launch_task_id: Optional[str]
    operation_task_id: Optional[str]
    launch_operation_id: Optional[str] = None
    operation_id: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentStatus:
    handle: SubagentHandle
    state: SubagentState
    updated_at: float
    diagnostic: Optional[str] = None
    audit_metadata: Optional[SubagentAuditMetadata] = None


@dataclasses.dataclass(frozen=True)
class SubagentTerminalState:
    handle: SubagentHandle
    state: SubagentState
    completed: bool
    timed_out: bool = False
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentCancelResult:
    accepted: bool
    already_terminal: bool = False
    unknown_handle: bool = False
    unsupported: bool = False
    state: SubagentState = SubagentState.UNKNOWN
    audit_metadata: Optional[SubagentAuditMetadata] = None


@dataclasses.dataclass(frozen=True)
class SubagentControlResult:
    api_contract_version: int
    disposition: SubagentControlDisposition
    accepted: bool
    state: SubagentState
    diagnostic: Optional[str] = None
    audit_metadata: Optional[SubagentAuditMetadata] = None


@dataclasses.dataclass(frozen=True)
class SubagentResult:
    handle: SubagentHandle
    terminal_state: SubagentState
    ready: bool
    summary: Optional[str] = None
    structured_payload: Optional[Mapping[str, Any]] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_classification: Optional[str] = None
    error_message: Optional[str] = None
    usage_metadata: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: _FrozenMapping({})
    )
    tool_execution_summary: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: _FrozenMapping({})
    )
    result_hash: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentCompletion:
    api_contract_version: int
    handle_serialization_version: int
    event_id: Optional[str]
    handle: SubagentHandle
    ready: bool
    terminal_state: Optional[SubagentState]
    result: Optional[SubagentResult]
    collected_at: Optional[float]
    diagnostic: Optional[str] = None
    audit_metadata: Optional[SubagentAuditMetadata] = None


@dataclasses.dataclass(frozen=True)
class SubagentReconnectResult:
    connected: bool
    state: SubagentState
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentLifecycleCapabilities:
    api_contract_version: int
    handle_serialization_version: int
    features: frozenset[str]
    providers_are_host_resolved: bool
    working_directory_supported: bool
    restart_recovery: str


@dataclasses.dataclass(frozen=True)
class _ResolvedLaunchRequest:
    base: SubagentLaunchRequest
    child_toolsets: Optional[list[str]]
    exact_toolsets: bool
    is_v2: bool
    provider: Optional[str] = None
    reasoning_config: Optional[Mapping[str, Any]] = None


@dataclasses.dataclass
class _Record:
    handle: SubagentHandle
    state: SubagentState
    updated_at: float
    agent: Any = None
    future: Optional[Future] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[SubagentResult] = None
    expiry_owner: Any = None
    on_expire: Optional[Callable[[SubagentHandle], None]] = None
    admission_lease: Any = None
    completion_event_id: Optional[str] = None
    collected_at: Optional[float] = None
    launch_task_id: Optional[str] = None
    operation_task_id: Optional[str] = None
    launch_operation_id: Optional[str] = None
    operation_id: Optional[str] = None


class _Registry:
    """Thread-safe terminal-retention registry; never returns live records."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.records: dict[str, _Record] = {}
        self.correlations: dict[tuple[Optional[str], str], Any] = {}


_REGISTRY = _Registry()
# Daemon worker pool: a wedged/abandoned child must never block interpreter
# exit at atexit-join time (same rationale as _run_single_child's timeout
# executor and the async-delegation registry pool).
from tools.daemon_pool import DaemonThreadPoolExecutor as _DaemonExecutor

_EXECUTOR = _DaemonExecutor(max_workers=8, thread_name_prefix="hermes-lifecycle")
_SECRET = secrets.token_bytes(32)
_ACTIVE_PARENT_AGENT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "hermes_subagent_lifecycle_parent", default=None
)


@contextmanager
def bind_subagent_parent(parent_agent: Any):
    """Bind the host-owned parent for the current agent turn."""
    token = _ACTIVE_PARENT_AGENT.set(parent_agent)
    try:
        yield
    finally:
        _ACTIVE_PARENT_AGENT.reset(token)


def get_active_subagent_parent() -> Any:
    """Return the parent bound to this execution context, if any."""
    return _ACTIVE_PARENT_AGENT.get()


class SubagentLifecycleService:
    """Stable public service returned by :attr:`PluginContext.subagent_lifecycle`.

    Running children are in-process only.  Completed results remain available
    until process exit; ``reconnect`` accurately reports that a serialized
    handle cannot reconnect after a restart instead of launching work again.
    """

    def __init__(self, parent_agent_resolver: Callable[[], Any]) -> None:
        self._parent_agent_resolver = parent_agent_resolver
        self._expiry_owner = object()
        self._record_expiry_observer: Optional[
            Callable[[SubagentHandle], None]
        ] = None

    def _bind_record_expiry_observer(
        self, observer: Callable[[SubagentHandle], None]
    ) -> None:
        """Bind one host-owned observer to records launched by this service."""
        if not callable(observer):
            raise TypeError("record expiry observer must be callable")
        with _REGISTRY.lock:
            self._record_expiry_observer = observer

    def _unbind_record_expiry_observer(self) -> None:
        """Detach this service's observer from current and future records."""
        with _REGISTRY.lock:
            self._record_expiry_observer = None
            for record in _REGISTRY.records.values():
                if record.expiry_owner is self._expiry_owner:
                    record.expiry_owner = None
                    record.on_expire = None

    def capabilities(self) -> SubagentLifecycleCapabilities:
        return SubagentLifecycleCapabilities(
            api_contract_version=LIFECYCLE_API_CONTRACT_VERSION,
            handle_serialization_version=PUBLIC_CONTRACT_VERSION,
            features=frozenset(
                {
                    "launch",
                    "status",
                    "wait",
                    "cancel",
                    "result",
                    "reconnect",
                    "list",
                    "steer",
                    "stop",
                    "collect",
                    "exact_toolsets",
                    "provider_routing",
                    "reasoning_override",
                    "native_read_only_transport_gate",
                    "route_catalog",
                    "route_assessment",
                    "root_execution_context",
                }
            ),
            providers_are_host_resolved=True,
            working_directory_supported=False,
            restart_recovery="unsupported",
        )

    def catalog_routes(self) -> SubagentRouteCatalog:
        """Return a complete bounded snapshot of authenticated native routes.

        Discovery is read-only and never constructs a child or consumes
        admission. Any unsafe, malformed, or oversized inventory collapses to
        a fixed incomplete receipt rather than exposing an unsafe partial
        catalog. Unrelated reference-only credential metadata is safely
        omitted because it does not prove an authenticated route.
        """
        parent = self._parent_agent_resolver()
        if parent is None:
            return _new_route_catalog(
                complete=False,
                routes=(),
                candidate_count=0,
                reason="CATALOG_UNAVAILABLE",
            )
        try:
            from tools.delegate_tool import _catalog_subagent_routes

            discovered = _catalog_subagent_routes(parent)
            routes = tuple(
                SubagentRouteIdentity(provider, model)
                for provider, model in discovered
            )
        except OverflowError:
            return _new_route_catalog(
                complete=False,
                routes=(),
                candidate_count=0,
                reason="CATALOG_INCOMPLETE",
            )
        except Exception:
            return _new_route_catalog(
                complete=False,
                routes=(),
                candidate_count=0,
                reason="CATALOG_UNAVAILABLE",
            )
        return _new_route_catalog(
            complete=True,
            routes=routes,
            candidate_count=len(routes),
            reason="COMPLETE",
        )

    def assess_route(self, provider: str, model: str) -> SubagentRouteAssessment:
        """Resolve and assess one exact-empty route without launching work."""
        requested = SubagentRouteIdentity(provider, model)
        parent = self._parent_agent_resolver()
        if parent is None:
            return _new_route_assessment(
                route=requested,
                eligible=False,
                reason="ROUTE_UNAVAILABLE",
                transport="unavailable",
                authenticated=False,
                agent_capable=False,
                exact_empty_model_tools=False,
                mutation_evidence_complete=False,
                independent_mutation_channels=frozenset(),
                hermes_model_tool_count=0,
            )
        try:
            from tools.delegate_tool import (
                _assess_native_read_only_route,
                _resolve_subagent_route,
            )

            resolved = _resolve_subagent_route(
                provider=requested.provider,
                model=requested.model,
                parent_agent=parent,
                exact_empty=True,
            )
            receipt = _assess_native_read_only_route(resolved)
            route = SubagentRouteIdentity(receipt.provider, receipt.model)
            channels = frozenset(receipt.independent_mutation_channels)
            tool_count = receipt.hermes_model_tool_count
            reason = receipt.reason
            eligible = bool(receipt.eligible)
            transport = receipt.transport
            exact_empty_model_tools = receipt.hermes_model_tools_empty
            mutation_evidence_complete = receipt.mutation_evidence_complete
        except Exception:
            return _new_route_assessment(
                route=requested,
                eligible=False,
                reason="ROUTE_UNAVAILABLE",
                transport="unavailable",
                authenticated=False,
                agent_capable=False,
                exact_empty_model_tools=False,
                mutation_evidence_complete=False,
                independent_mutation_channels=frozenset(),
                hermes_model_tool_count=0,
            )
        return _new_route_assessment(
            route=route,
            eligible=eligible,
            reason=reason,
            transport=transport,
            authenticated=True,
            agent_capable=True,
            exact_empty_model_tools=exact_empty_model_tools,
            mutation_evidence_complete=mutation_evidence_complete,
            independent_mutation_channels=channels,
            hermes_model_tool_count=tool_count,
        )

    def _record_audit_metadata(
        self,
        handle: SubagentHandle,
        *,
        launch_task_id: Any = _AUDIT_UNSET,
        operation_task_id: Any = _AUDIT_UNSET,
        launch_operation_id: Any = _AUDIT_UNSET,
        operation_id: Any = _AUDIT_UNSET,
    ) -> Optional[SubagentAuditMetadata]:
        """Attach host-owned task provenance without making it authority."""
        record = self._record(handle)
        if record is None:
            return None
        with _REGISTRY.lock:
            if _REGISTRY.records.get(handle.subagent_id) is not record:
                return None
            if launch_task_id is not _AUDIT_UNSET and record.launch_task_id is None:
                record.launch_task_id = (
                    str(launch_task_id)[:256] if launch_task_id is not None else None
                )
            if operation_task_id is not _AUDIT_UNSET:
                record.operation_task_id = (
                    str(operation_task_id)[:256]
                    if operation_task_id is not None
                    else None
                )
            if (
                launch_operation_id is not _AUDIT_UNSET
                and record.launch_operation_id is None
            ):
                record.launch_operation_id = (
                    str(launch_operation_id)[:64]
                    if launch_operation_id is not None
                    else None
                )
            if operation_id is not _AUDIT_UNSET:
                record.operation_id = (
                    str(operation_id)[:64] if operation_id is not None else None
                )
            return SubagentAuditMetadata(
                record.launch_task_id,
                record.operation_task_id,
                record.launch_operation_id,
                record.operation_id,
            )

    def launch(
        self, request: SubagentLaunchRequest | SubagentLaunchRequestV2
    ) -> SubagentHandle:
        parent = self._parent_agent_resolver()
        if parent is None:
            raise SubagentLifecycleError(
                "No active Hermes parent session is available."
            )
        resolved_request = self._resolve_request(request, parent)
        request = resolved_request.base
        from tools.delegate_tool import (
            _assess_native_read_only_route,
            _build_child_preserving_parent_tools,
            _get_max_concurrent_children,
            _get_max_spawn_depth,
            _resolve_subagent_route,
            _verify_native_read_only_child,
            DEFAULT_MAX_ITERATIONS,
        )
        from tools.delegation_admission import (
            try_admit_background_unit,
        )
        parent_session_id = str(getattr(parent, "session_id", "") or "") or None
        if request.parent_session_id and request.parent_session_id != parent_session_id:
            raise SubagentLifecycleError(
                "parent_session_id does not match the active session."
            )

        route = None
        route_resolution_failed = False
        exact_empty = (
            resolved_request.exact_toolsets
            and not resolved_request.child_toolsets
        )
        if resolved_request.is_v2 and (
            resolved_request.provider is not None or request.model is not None
        ):
            try:
                route = _resolve_subagent_route(
                    provider=resolved_request.provider,
                    model=request.model,
                    parent_agent=parent,
                    exact_empty=exact_empty,
                )
            except Exception:
                route_resolution_failed = True
        if route_resolution_failed:
            # Raise after leaving the resolver's exception handler so neither
            # __cause__ nor implicit __context__ retains profile diagnostics.
            raise SubagentLifecycleError(
                "Requested provider/model route is unavailable."
            )

        read_only_receipt = None
        if exact_empty:
            effective_route = route or {
                "provider": getattr(parent, "provider", None),
                "model": getattr(parent, "model", None),
                "api_mode": getattr(parent, "api_mode", None),
                "command": getattr(parent, "acp_command", None),
                "args": list(getattr(parent, "acp_args", None) or []),
            }
            read_only_receipt = _assess_native_read_only_route(effective_route)
            if not read_only_receipt.eligible:
                raise SubagentLifecycleError(
                    "Native read-only transport is unavailable."
                )

        # ThreadPoolExecutor does not propagate ContextVars. Capture the
        # canonical HERMES_HOME and secret scope now so the entire child turn,
        # including credential refreshes, remains anchored to launch profile.
        launch_context = contextvars.copy_context()
        correlation_key = (parent_session_id, request.correlation_id or "")
        correlation_owner = None
        with _REGISTRY.lock:
            self._cleanup_locked()
            if request.correlation_id and correlation_key in _REGISTRY.correlations:
                raise SubagentLifecycleError(
                    "Duplicate correlation_id for this parent session."
                )
            if request.correlation_id:
                correlation_owner = object()
                _REGISTRY.correlations[correlation_key] = correlation_owner

        try:
            admission = try_admit_background_unit(
                _get_max_concurrent_children(),
                enforce_pause=True,
                parent_depth=int(getattr(parent, "_delegate_depth", 0) or 0),
                max_spawn_depth=_get_max_spawn_depth(),
                batch_size=1,
                batch_enabled=False,
            )
        except Exception:
            if request.correlation_id:
                with _REGISTRY.lock:
                    if _REGISTRY.correlations.get(correlation_key) is correlation_owner:
                        _REGISTRY.correlations.pop(correlation_key, None)
            raise
        if not admission.admitted:
            if request.correlation_id:
                with _REGISTRY.lock:
                    if _REGISTRY.correlations.get(correlation_key) is correlation_owner:
                        _REGISTRY.correlations.pop(correlation_key, None)
            raise SubagentLifecycleError(admission.rejection_code or "CAPACITY_REACHED")
        lease = admission.lease
        record = None
        child = None
        launch_failure_message = None
        try:
            # Delegate construction remains internal so plugin code never imports
            # private delegation helpers or manipulates the active-child registry.
            build_kwargs = dict(
                task_index=0,
                goal=request.goal,
                context=request.context,
                toolsets=resolved_request.child_toolsets,
                exact_toolsets=resolved_request.exact_toolsets,
                model=(route["model"] if route is not None else request.model),
                max_iterations=DEFAULT_MAX_ITERATIONS,
                task_count=1,
                parent_agent=parent,
                role=request.role,
            )
            if route is not None:
                build_kwargs.update(
                    override_provider=route["provider"],
                    override_base_url=route["base_url"],
                    override_api_key=route["api_key"],
                    override_api_mode=route["api_mode"],
                    override_request_overrides=route["request_overrides"],
                    override_max_tokens=route["max_output_tokens"],
                    override_acp_command=route["command"],
                    override_acp_args=route["args"],
                    authoritative_route_overrides=True,
                )
            if resolved_request.reasoning_config is not None:
                build_kwargs["override_reasoning_config"] = dict(
                    resolved_request.reasoning_config
                )
            child = _build_child_preserving_parent_tools(**build_kwargs)
            if read_only_receipt is not None and not _verify_native_read_only_child(
                child, read_only_receipt
            ):
                raise SubagentLifecycleError(
                    "Native read-only transport is unavailable."
                )
            subagent_id = str(getattr(child, "_subagent_id", "") or "")
            if not subagent_id:
                raise SubagentLifecycleError("Hermes failed to assign a child identity.")
            created = time.time()
            handle = SubagentHandle(
                PUBLIC_CONTRACT_VERSION,
                subagent_id,
                parent_session_id,
                request.correlation_id,
                created,
                getattr(child, "provider", None),
                getattr(child, "model", None),
                getattr(child, "_delegate_role", request.role),
                int(getattr(child, "_delegate_depth", 1) or 1),
                self._capability(subagent_id, parent_session_id, created),
            )
            with _REGISTRY.lock:
                record = _Record(
                    handle,
                    SubagentState.PENDING,
                    created,
                    agent=child,
                    expiry_owner=self._expiry_owner,
                    on_expire=self._record_expiry_observer,
                    admission_lease=lease,
                )
                _REGISTRY.records[subagent_id] = record
                if request.correlation_id:
                    if _REGISTRY.correlations.get(correlation_key) is not correlation_owner:
                        raise SubagentLifecycleError(
                            "Correlation reservation ownership was lost."
                        )
                    _REGISTRY.correlations[correlation_key] = subagent_id
                    correlation_owner = subagent_id
            record.future = _EXECUTOR.submit(
                launch_context.run, self._run, record, request.goal, parent
            )
            return handle
        except Exception as exc:
            trusted_failure_messages = {
                "Native read-only transport is unavailable.",
                "Hermes failed to assign a child identity.",
                "Correlation reservation ownership was lost.",
            }
            if (
                isinstance(exc, SubagentLifecycleError)
                and str(exc) in trusted_failure_messages
            ):
                launch_failure_message = str(exc)
            elif route is not None:
                launch_failure_message = (
                    "Requested provider/model route could not be launched."
                )
            else:
                launch_failure_message = "Hermes failed to launch subagent."
            if record is not None:
                with _REGISTRY.lock:
                    if _REGISTRY.records.get(record.handle.subagent_id) is record:
                        _REGISTRY.records.pop(record.handle.subagent_id, None)
                    if (
                        request.correlation_id
                        and _REGISTRY.correlations.get(correlation_key)
                        == correlation_owner
                    ):
                        _REGISTRY.correlations.pop(correlation_key, None)
                    record.admission_lease = None
            elif request.correlation_id:
                with _REGISTRY.lock:
                    if _REGISTRY.correlations.get(correlation_key) is correlation_owner:
                        _REGISTRY.correlations.pop(correlation_key, None)
            if lease is not None:
                lease.release()
            if child is not None:
                try:
                    active_children = getattr(parent, "_active_children", None)
                    if active_children is not None:
                        active_lock = getattr(parent, "_active_children_lock", None)
                        if active_lock is not None:
                            with active_lock:
                                if child in active_children:
                                    active_children.remove(child)
                        elif child in active_children:
                            active_children.remove(child)
                    close = getattr(child, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
        # Keep this raise outside the active exception handler. Public plugin
        # callers and registry logging must never traverse raw build failures
        # through either __cause__ or implicit __context__.
        raise SubagentLifecycleError(launch_failure_message)

    def status(self, handle: SubagentHandle) -> SubagentStatus:
        record = self._record(handle)
        if record is None:
            return SubagentStatus(
                handle, SubagentState.UNKNOWN, time.time(), "UNKNOWN_HANDLE"
            )
        with _REGISTRY.lock:
            return SubagentStatus(record.handle, record.state, record.updated_at)

    def list(self) -> tuple[SubagentStatus, ...]:
        parent = self._parent_agent_resolver()
        parent_session_id = str(getattr(parent, "session_id", "") or "") or None
        with _REGISTRY.lock:
            self._cleanup_locked()
            records = sorted(
                (
                    record
                    for record in _REGISTRY.records.values()
                    if record.expiry_owner is self._expiry_owner
                    and record.handle.parent_session_id == parent_session_id
                ),
                key=lambda record: (record.handle.created_at, record.handle.subagent_id),
            )
            return tuple(
                SubagentStatus(record.handle, record.state, record.updated_at)
                for record in records
            )

    def wait(
        self, handle: SubagentHandle, *, timeout_seconds: Optional[float] = None
    ) -> SubagentTerminalState:
        record = self._record(handle)
        if record is None:
            return SubagentTerminalState(
                handle, SubagentState.UNKNOWN, True, diagnostic="UNKNOWN_HANDLE"
            )
        future = record.future
        if future is not None:
            try:
                future.result(timeout=timeout_seconds)
            except TimeoutError:
                return SubagentTerminalState(record.handle, record.state, False, True)
            except Exception:
                pass
        with _REGISTRY.lock:
            return SubagentTerminalState(
                record.handle, record.state, record.result is not None
            )

    def steer(self, handle: SubagentHandle, text: str) -> SubagentControlResult:
        if not isinstance(text, str) or not text.strip() or len(text) > _MAX_CONTEXT_CHARS:
            raise SubagentLifecycleError(
                "steer text must be a non-empty string of at most 32000 characters."
            )
        record = self._record(handle)
        if record is None:
            return SubagentControlResult(
                LIFECYCLE_API_CONTRACT_VERSION,
                SubagentControlDisposition.UNKNOWN_HANDLE,
                False,
                SubagentState.UNKNOWN,
                "UNKNOWN_HANDLE",
            )
        with _REGISTRY.lock:
            if (
                _REGISTRY.records.get(record.handle.subagent_id) is not record
                or record.expiry_owner is not self._expiry_owner
            ):
                return SubagentControlResult(
                    LIFECYCLE_API_CONTRACT_VERSION,
                    SubagentControlDisposition.UNKNOWN_HANDLE,
                    False,
                    SubagentState.UNKNOWN,
                    "UNKNOWN_HANDLE",
                )
            if record.result is not None:
                return SubagentControlResult(
                    LIFECYCLE_API_CONTRACT_VERSION,
                    SubagentControlDisposition.TERMINAL,
                    False,
                    record.state,
                    "ALREADY_TERMINAL",
                )
            agent = record.agent
            state = record.state
        steer = getattr(agent, "steer", None)
        if not callable(steer):
            return SubagentControlResult(
                LIFECYCLE_API_CONTRACT_VERSION,
                SubagentControlDisposition.UNSUPPORTED,
                False,
                state,
                "STEER_UNSUPPORTED",
            )
        try:
            accepted = bool(steer(text.strip()))
        except Exception:
            return SubagentControlResult(
                LIFECYCLE_API_CONTRACT_VERSION,
                SubagentControlDisposition.UNSUPPORTED,
                False,
                state,
                "STEER_UNSUPPORTED",
            )
        with _REGISTRY.lock:
            if (
                _REGISTRY.records.get(record.handle.subagent_id) is not record
                or record.expiry_owner is not self._expiry_owner
            ):
                return SubagentControlResult(
                    LIFECYCLE_API_CONTRACT_VERSION,
                    SubagentControlDisposition.UNKNOWN_HANDLE,
                    False,
                    SubagentState.UNKNOWN,
                    "UNKNOWN_HANDLE",
                )
            if record.result is not None or record.agent is not agent:
                return SubagentControlResult(
                    LIFECYCLE_API_CONTRACT_VERSION,
                    SubagentControlDisposition.MISSED,
                    False,
                    record.state,
                    "STEER_MISSED",
                )
            return SubagentControlResult(
                LIFECYCLE_API_CONTRACT_VERSION,
                SubagentControlDisposition.QUEUED
                if accepted
                else SubagentControlDisposition.MISSED,
                accepted,
                record.state,
                None if accepted else "STEER_MISSED",
            )

    def cancel(self, handle: SubagentHandle, *, reason: str) -> SubagentCancelResult:
        record = self._record(handle)
        if record is None:
            return SubagentCancelResult(False, unknown_handle=True)
        with _REGISTRY.lock:
            if record.result is not None:
                return SubagentCancelResult(
                    False, already_terminal=True, state=record.state
                )
            agent = record.agent
            record.state = SubagentState.CANCEL_REQUESTED
            record.updated_at = time.time()
        if agent is None:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        try:
            accepted = request_hard_interrupt(
                agent,
                f"Lifecycle cancellation requested: {reason[:500]}",
                tool_reason="subagent cancellation requested",
            )
        except Exception:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        if not accepted:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        return SubagentCancelResult(True, state=SubagentState.CANCEL_REQUESTED)

    def stop(self, handle: SubagentHandle, *, reason: str) -> SubagentCancelResult:
        return self.cancel(handle, reason=reason)

    def result(self, handle: SubagentHandle) -> SubagentResult:
        record = self._record(handle)
        if record is None:
            return SubagentResult(
                handle,
                SubagentState.UNKNOWN,
                False,
                error_classification="UNKNOWN_HANDLE",
            )
        with _REGISTRY.lock:
            if record.result is not None:
                return record.result
            return SubagentResult(
                record.handle, record.state, False, error_classification="NOT_READY"
            )

    def collect(self, handle: SubagentHandle) -> SubagentCompletion:
        record = self._record(handle)
        if record is None:
            return SubagentCompletion(
                LIFECYCLE_API_CONTRACT_VERSION,
                PUBLIC_CONTRACT_VERSION,
                None,
                handle,
                False,
                None,
                None,
                None,
                "UNKNOWN_HANDLE",
            )
        with _REGISTRY.lock:
            if record.result is None:
                return SubagentCompletion(
                    LIFECYCLE_API_CONTRACT_VERSION,
                    PUBLIC_CONTRACT_VERSION,
                    None,
                    record.handle,
                    False,
                    None,
                    None,
                    None,
                )
            if record.collected_at is None:
                record.collected_at = time.time()
                event_material = (
                    f"{record.handle.subagent_id}|"
                    f"{record.result.result_hash or ''}|"
                    f"{record.result.completed_at or 0:.6f}"
                ).encode()
                record.completion_event_id = (
                    "subagent-completion-"
                    + hashlib.sha256(event_material).hexdigest()
                )
            return SubagentCompletion(
                LIFECYCLE_API_CONTRACT_VERSION,
                PUBLIC_CONTRACT_VERSION,
                record.completion_event_id,
                record.handle,
                True,
                record.result.terminal_state,
                record.result,
                record.collected_at,
            )

    def reconnect(self, handle: SubagentHandle) -> SubagentReconnectResult:
        record = self._record(handle)
        if record is None:
            return SubagentReconnectResult(
                False, SubagentState.UNKNOWN, "RECONNECT_UNAVAILABLE"
            )
        with _REGISTRY.lock:
            return SubagentReconnectResult(True, record.state)

    def _record(self, handle: SubagentHandle) -> Optional[_Record]:
        if (
            not isinstance(handle, SubagentHandle)
            or type(handle.contract_version) is not int
            or handle.contract_version != PUBLIC_CONTRACT_VERSION
        ):
            return None
        if (
            not isinstance(handle.subagent_id, str)
            or not handle.subagent_id
            or (
                handle.parent_session_id is not None
                and not isinstance(handle.parent_session_id, str)
            )
            or (
                handle.correlation_id is not None
                and not isinstance(handle.correlation_id, str)
            )
            or isinstance(handle.created_at, bool)
            or not isinstance(handle.created_at, (int, float))
            or not math.isfinite(handle.created_at)
            or (handle.provider is not None and not isinstance(handle.provider, str))
            or (handle.model is not None and not isinstance(handle.model, str))
            or not isinstance(handle.role, str)
            or type(handle.depth) is not int
            or not isinstance(handle.capability, str)
        ):
            return None
        if not hmac.compare_digest(
            handle.capability,
            self._capability(
                handle.subagent_id, handle.parent_session_id, handle.created_at
            ),
        ):
            return None
        parent = self._parent_agent_resolver()
        active_parent_id = str(getattr(parent, "session_id", "") or "") or None
        if active_parent_id != handle.parent_session_id:
            return None
        with _REGISTRY.lock:
            # Retention is authoritative at every public lookup, not only as
            # a side effect of a later launch. The observer runs while the
            # registry owns removal so host ownership metadata co-expires.
            self._cleanup_locked()
            record = _REGISTRY.records.get(handle.subagent_id)
            if (
                record is None
                or record.expiry_owner is not self._expiry_owner
                or record.handle != handle
            ):
                return None
            return record

    @staticmethod
    def _cleanup_locked() -> None:
        """Retain terminal snapshots for a bounded period, never live work."""
        cutoff = time.time() - _TERMINAL_RETENTION_SECONDS
        expired = [
            subagent_id
            for subagent_id, record in _REGISTRY.records.items()
            if record.result is not None
            and record.completed_at is not None
            and record.completed_at < cutoff
        ]
        for subagent_id in expired:
            record = _REGISTRY.records.pop(subagent_id)
            if record.handle.correlation_id:
                correlation_key = (
                    record.handle.parent_session_id,
                    record.handle.correlation_id,
                )
                if (
                    _REGISTRY.correlations.get(correlation_key)
                    == record.handle.subagent_id
                ):
                    _REGISTRY.correlations.pop(correlation_key, None)
            if record.on_expire is not None:
                try:
                    record.on_expire(record.handle)
                except Exception:
                    # Retention cleanup is authoritative. An optional host
                    # observer must never keep an expired lifecycle record
                    # alive or make an unrelated launch/status operation fail.
                    pass

    def _run(self, record: _Record, goal: str, parent: Any) -> None:
        try:
            self._run_admitted(record, goal, parent)
        finally:
            with _REGISTRY.lock:
                lease = record.admission_lease
                record.admission_lease = None
            if lease is not None:
                lease.release()

    def _run_admitted(self, record: _Record, goal: str, parent: Any) -> None:
        with _REGISTRY.lock:
            if record.state is not SubagentState.CANCEL_REQUESTED:
                record.state = SubagentState.RUNNING
            record.started_at = time.time()
            record.updated_at = record.started_at
        try:
            from tools.delegate_tool import _run_child_lifecycle

            raw = _run_child_lifecycle(0, goal, record.agent, parent)
            status = (
                str(raw.get("status", "error")) if isinstance(raw, dict) else "error"
            )
            if status == "completed":
                state = SubagentState.SUCCEEDED
            elif status == "interrupted":
                state = (
                    SubagentState.CANCELLED
                    if record.state == SubagentState.CANCEL_REQUESTED
                    else SubagentState.INTERRUPTED
                )
            else:
                state = SubagentState.FAILED
            error_classification = {
                "interrupted": "CHILD_INTERRUPTED",
                "timeout": "CHILD_TIMEOUT",
                "timed_out": "CHILD_TIMEOUT",
            }.get(status, "CHILD_FAILED")
            error_message = {
                "interrupted": "Child was interrupted.",
                "timeout": "Child timed out.",
                "timed_out": "Child timed out.",
            }.get(status, "Child reported failure.")
            summary = raw.get("summary") if isinstance(raw, dict) else None
            summary = str(summary)[:_MAX_RESULT_CHARS] if summary is not None else None
            structured_payload = (
                _freeze_structured_payload(raw.get("structured_payload"))
                if isinstance(raw, dict)
                and isinstance(raw.get("structured_payload"), Mapping)
                else None
            )
            result = SubagentResult(
                record.handle,
                state,
                True,
                summary=summary,
                structured_payload=structured_payload,
                completed_at=time.time(),
                started_at=record.started_at,
                error_classification=None
                if state == SubagentState.SUCCEEDED
                else error_classification,
                error_message=None
                if state == SubagentState.SUCCEEDED
                else error_message,
                usage_metadata=_deep_freeze({
                    "api_calls": _bounded_api_calls(raw.get("api_calls", 0))
                })
                if isinstance(raw, dict)
                else _FrozenMapping({}),
                tool_execution_summary=_deep_freeze({
                    "duration_seconds": _bounded_duration_seconds(
                        raw.get("duration_seconds", 0)
                    )
                })
                if isinstance(raw, dict)
                else _FrozenMapping({}),
            )
        except Exception:
            result = SubagentResult(
                record.handle,
                SubagentState.FAILED,
                True,
                started_at=record.started_at,
                completed_at=time.time(),
                error_classification="INTERNAL_ERROR",
                error_message="Child execution failed.",
            )
        result = dataclasses.replace(
            result,
            result_hash=hashlib.sha256(
                json.dumps(
                    _canonical_hash_value(result),
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode()
            ).hexdigest(),
        )
        with _REGISTRY.lock:
            record.agent = None
            record.result = result
            record.state = result.terminal_state
            record.completed_at = result.completed_at
            record.updated_at = result.completed_at or time.time()

    @staticmethod
    def _capability(
        subagent_id: str, parent_session_id: Optional[str], created_at: float
    ) -> str:
        value = f"{subagent_id}|{parent_session_id or ''}|{created_at:.6f}".encode()
        return hmac.new(_SECRET, value, hashlib.sha256).hexdigest()

    @staticmethod
    def _validate_request(request: SubagentLaunchRequest, parent: Any) -> None:
        if (
            not isinstance(request, SubagentLaunchRequest)
            or not isinstance(request.goal, str)
            or not request.goal.strip()
            or len(request.goal) > _MAX_GOAL_CHARS
        ):
            raise SubagentLifecycleError(
                "goal must be a non-empty string of at most 16000 characters."
            )
        if request.context is not None and (
            not isinstance(request.context, str)
            or len(request.context) > _MAX_CONTEXT_CHARS
        ):
            raise SubagentLifecycleError(
                "context must be a string of at most 32000 characters."
            )
        if request.role not in {"leaf", "orchestrator"}:
            raise SubagentLifecycleError("role must be 'leaf' or 'orchestrator'.")
        if request.timeout_seconds is not None:
            raise SubagentLifecycleError(
                "Per-launch timeout is not supported; configure delegation timeout explicitly."
            )
        if request.working_directory is not None:
            raise SubagentLifecycleError(
                "working_directory is not supported because Hermes delegates use isolated task environments."
            )
        if request.blocked_tools:
            raise SubagentLifecycleError(
                "Per-tool blocking is not supported; use allowed_toolsets. Hermes always blocks unsafe child tools."
            )
        try:
            metadata_bytes = len(
                json.dumps(dict(request.metadata), sort_keys=True).encode()
            )
        except (TypeError, ValueError) as exc:
            raise SubagentLifecycleError("metadata must be JSON-serializable.") from exc
        if metadata_bytes > _MAX_METADATA_BYTES:
            raise SubagentLifecycleError("metadata exceeds 8192 bytes.")
        if request.allowed_toolsets:
            from toolsets import TOOLSETS

            unknown = set(request.allowed_toolsets) - set(TOOLSETS)
            if unknown:
                raise SubagentLifecycleError(
                    f"Unknown toolsets: {', '.join(sorted(unknown))}."
                )
            enabled = getattr(parent, "enabled_toolsets", None)
            if enabled is not None and not set(request.allowed_toolsets).issubset(
                set(enabled)
            ):
                raise SubagentLifecycleError(
                    "Requested toolsets would broaden parent permissions."
                )

    @classmethod
    def _resolve_request(
        cls,
        request: SubagentLaunchRequest | SubagentLaunchRequestV2,
        parent: Any,
    ) -> _ResolvedLaunchRequest:
        if isinstance(request, SubagentLaunchRequest):
            cls._validate_request(request, parent)
            return _ResolvedLaunchRequest(
                base=request,
                child_toolsets=(
                    list(request.allowed_toolsets) if request.allowed_toolsets else None
                ),
                exact_toolsets=False,
                is_v2=False,
            )
        if not isinstance(request, SubagentLaunchRequestV2):
            raise SubagentLifecycleError("Unsupported lifecycle launch request.")
        if (
            type(request.api_contract_version) is not int
            or request.api_contract_version
            not in _SUPPORTED_LAUNCH_REQUEST_API_CONTRACT_VERSIONS
        ):
            raise SubagentLifecycleError("Unsupported lifecycle API contract version.")
        if not isinstance(request.base, SubagentLaunchRequest):
            raise SubagentLifecycleError("base must be a SubagentLaunchRequest.")
        if request.base.allowed_toolsets is not None:
            raise SubagentLifecycleError(
                "V2 base.allowed_toolsets must be None; use toolset_mode."
            )
        if request.toolset_mode not in {"inherit", "exact"}:
            raise SubagentLifecycleError("toolset_mode must be 'inherit' or 'exact'.")
        if not isinstance(request.exact_toolsets, tuple) or not all(
            isinstance(value, str) and value for value in request.exact_toolsets
        ):
            raise SubagentLifecycleError(
                "exact_toolsets must be a tuple of non-empty toolset names."
            )
        if request.toolset_mode == "inherit" and request.exact_toolsets:
            raise SubagentLifecycleError(
                "inherit mode cannot declare exact_toolsets."
            )
        cls._validate_request(request.base, parent)
        reasoning_config = None
        if request.reasoning_effort is not None:
            from hermes_constants import parse_reasoning_effort

            reasoning_config = parse_reasoning_effort(request.reasoning_effort)
            if reasoning_config is None:
                raise SubagentLifecycleError(
                    "reasoning_effort must be false, 'none', or a supported effort identifier."
                )
        if request.toolset_mode == "inherit":
            return _ResolvedLaunchRequest(
                base=request.base,
                child_toolsets=None,
                exact_toolsets=False,
                is_v2=True,
                provider=request.provider,
                reasoning_config=reasoning_config,
            )

        from toolsets import TOOLSETS

        unknown = set(request.exact_toolsets) - set(TOOLSETS)
        if unknown:
            raise SubagentLifecycleError(
                f"Unknown toolsets: {', '.join(sorted(unknown))}."
            )
        if "delegation" in request.exact_toolsets:
            raise SubagentLifecycleError(
                "Delegation cannot be granted through exact toolsets."
            )
        enabled = getattr(parent, "enabled_toolsets", None)
        if enabled is not None:
            from tools.delegate_tool import _expand_parent_toolsets

            expanded_parent = _expand_parent_toolsets(set(enabled))
            if not set(request.exact_toolsets).issubset(expanded_parent):
                raise SubagentLifecycleError(
                    "Requested toolsets would broaden parent permissions."
                )
        return _ResolvedLaunchRequest(
            base=request.base,
            child_toolsets=list(request.exact_toolsets),
            exact_toolsets=True,
            is_v2=True,
            provider=request.provider,
            reasoning_config=reasoning_config,
        )
