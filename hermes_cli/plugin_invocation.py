"""Host-minted invocation identity for explicitly opted-in plugin tools.

The public lifecycle facade is stateless. Host services, authority, and session
binding live in a weak identity registry, so copying or reconstructing an
object with the same apparent value never grants authority.
"""

from __future__ import annotations

import dataclasses
import contextvars
import secrets
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from agent.subagent_lifecycle import (
    LIFECYCLE_API_CONTRACT_VERSION,
    PUBLIC_CONTRACT_VERSION,
    SubagentCancelResult,
    SubagentCompletion,
    SubagentControlDisposition,
    SubagentControlResult,
    SubagentHandle,
    SubagentLaunchRequest,
    SubagentLaunchRequestV2,
    SubagentLifecycleCapabilities,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentReconnectResult,
    SubagentResult,
    SubagentRouteAssessment,
    SubagentRouteCatalog,
    SubagentState,
    SubagentStatus,
    SubagentTerminalState,
    _TERMINAL_RETENTION_SECONDS,
)

PLUGIN_INVOCATION_CONTRACT_VERSION = 2
_TERMINAL_STATES = frozenset({
    SubagentState.SUCCEEDED,
    SubagentState.FAILED,
    SubagentState.INTERRUPTED,
    SubagentState.CANCELLED,
})


@dataclasses.dataclass(frozen=True)
class _PluginAuthority:
    plugin_id: str
    canonical_profile_key: str
    manager_scope_key: str


@dataclasses.dataclass
class _OwnedHandle:
    handle: SubagentHandle
    terminal_at: Optional[float] = None


class _OwnerStore:
    """Per-plugin bounded ownership metadata shared by derived facades."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.owners: dict[str, _OwnedHandle] = {}
        self.revoked = False

    def record(self, handle: SubagentHandle) -> None:
        with self.lock:
            self._prune_locked(time.time())
            if not self.revoked:
                self.owners[handle.subagent_id] = _OwnedHandle(handle)

    def authorize(self, handle: SubagentHandle, session_id: Optional[str]) -> bool:
        with self.lock:
            self._prune_locked(time.time())
            owner = self.owners.get(getattr(handle, "subagent_id", ""))
            return bool(
                not self.revoked
                and owner is not None
                and owner.handle == handle
                and owner.handle.parent_session_id == session_id
            )

    def mark_terminal(self, handle: SubagentHandle, completed_at: float) -> None:
        with self.lock:
            owner = self.owners.get(handle.subagent_id)
            if owner is not None and owner.handle == handle:
                owner.terminal_at = completed_at

    def forget(self, handle: SubagentHandle) -> None:
        with self.lock:
            owner = self.owners.get(getattr(handle, "subagent_id", ""))
            if owner is not None and owner.handle == handle:
                self.owners.pop(handle.subagent_id, None)

    def handles_for_session(
        self, session_id: Optional[str]
    ) -> tuple[SubagentHandle, ...]:
        with self.lock:
            self._prune_locked(time.time())
            if self.revoked:
                return ()
            return tuple(
                owner.handle
                for owner in self.owners.values()
                if owner.handle.parent_session_id == session_id
            )

    def revoke(self) -> None:
        with self.lock:
            self.revoked = True
            self.owners.clear()

    def is_revoked(self) -> bool:
        with self.lock:
            return self.revoked

    def _prune_locked(self, now: float) -> None:
        expired = [
            subagent_id
            for subagent_id, owner in self.owners.items()
            if owner.terminal_at is not None
            and now - owner.terminal_at > _TERMINAL_RETENTION_SECONDS
        ]
        for subagent_id in expired:
            self.owners.pop(subagent_id, None)


class _RootAuthority:
    """One root plugin authority plus its operation/unload admission gate.

    The condition never nests with the lifecycle service registry lock.  A
    lifecycle operation holds an admission token while it calls the service;
    launch additionally publishes ownership before releasing its token.
    Unload first closes admission, then waits without holding any binding,
    owner, or lifecycle lock.
    """

    def __init__(self, authority: _PluginAuthority, owners: _OwnerStore) -> None:
        self.authority = authority
        self.owners = owners
        self._condition = threading.Condition(threading.RLock())
        self._revoking = False
        self._admitted_operations = 0

    def admit_operation(self) -> bool:
        with self._condition:
            if self._revoking:
                return False
            self._admitted_operations += 1
            return True

    def complete_operation(self) -> None:
        with self._condition:
            self._admitted_operations -= 1
            self._condition.notify_all()

    def admit_launch(self) -> bool:
        return self.admit_operation()

    def complete_launch(self, handle: Optional[SubagentHandle]) -> None:
        with self._condition:
            # Publishing ownership is part of the admitted launch.  Unload
            # cannot revoke the store until this critical section completes.
            if handle is not None:
                self.owners.record(handle)
            self._admitted_operations -= 1
            self._condition.notify_all()

    def begin_revoke_and_wait(self) -> None:
        with self._condition:
            self._revoking = True
            while self._admitted_operations:
                self._condition.wait()

    def can_mint(self) -> bool:
        with self._condition:
            return not self._revoking


@dataclasses.dataclass(frozen=True)
class _Binding:
    service: SubagentLifecycleService
    root: _RootAuthority
    parent_resolver: Callable[[], Any]
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    operation_id: Optional[str] = None


class BoundSubagentLifecycle:
    """Stateless, host-minted authority facade over the lifecycle service."""

    __slots__ = ("__weakref__",)

    def __new__(cls, *_args, **_kwargs):
        raise TypeError("BoundSubagentLifecycle objects are minted by Hermes.")

    def launch(
        self, request: SubagentLaunchRequest | SubagentLaunchRequestV2
    ) -> SubagentHandle:
        binding = _binding_for(self)
        if (
            binding is None
            or not _active_authority_matches(binding)
            or not binding.root.admit_launch()
        ):
            raise SubagentLifecycleError("Subagent lifecycle authority is unavailable.")
        handle = None
        try:
            active_session = _active_session(binding)
            if binding.session_id is not None and active_session != binding.session_id:
                raise SubagentLifecycleError(
                    "Invocation session does not match the active Hermes session."
                )
            with _binding_profile_scope(binding):
                handle = binding.service.launch(request)
            binding.service._record_audit_metadata(
                handle,
                launch_task_id=binding.task_id,
                operation_task_id=binding.task_id,
                launch_operation_id=_operation_id(binding),
                operation_id=_operation_id(binding),
            )
            return handle
        finally:
            binding.root.complete_launch(handle)

    def capabilities(self) -> SubagentLifecycleCapabilities:
        binding = _active_binding(self)
        if binding is None:
            raise SubagentLifecycleError("Subagent lifecycle authority is unavailable.")
        return binding.service.capabilities()

    def catalog_routes(self) -> SubagentRouteCatalog:
        with _admitted_active_binding(self) as binding:
            if binding is None:
                raise SubagentLifecycleError(
                    "Subagent lifecycle authority is unavailable."
                )
            with _binding_profile_scope(binding):
                return binding.service.catalog_routes()

    def assess_route(self, provider: str, model: str) -> SubagentRouteAssessment:
        with _admitted_active_binding(self) as binding:
            if binding is None:
                raise SubagentLifecycleError(
                    "Subagent lifecycle authority is unavailable."
                )
            with _binding_profile_scope(binding):
                return binding.service.assess_route(provider, model)

    def list(self) -> tuple[SubagentStatus, ...]:
        with _admitted_active_binding(self) as binding:
            if binding is None:
                return ()
            session_id = _expected_session(binding)
            owned = set(binding.root.owners.handles_for_session(session_id))
            return tuple(
                dataclasses.replace(
                    status,
                    audit_metadata=binding.service._record_audit_metadata(
                        status.handle,
                        operation_task_id=binding.task_id,
                        operation_id=_operation_id(binding),
                    ),
                )
                for status in binding.service.list()
                if status.handle in owned
            )

    def steer(self, handle: SubagentHandle, text: str) -> SubagentControlResult:
        with _admitted_authorized_binding(self, handle) as binding:
            if binding is None:
                return SubagentControlResult(
                    LIFECYCLE_API_CONTRACT_VERSION,
                    SubagentControlDisposition.WRONG_AUTHORITY,
                    False,
                    SubagentState.UNKNOWN,
                    "WRONG_AUTHORITY",
                )
            result = binding.service.steer(handle, text)
            _observe_state(binding, handle, result.state)
            return dataclasses.replace(
                result,
                audit_metadata=binding.service._record_audit_metadata(
                    handle,
                    operation_task_id=binding.task_id,
                    operation_id=_operation_id(binding),
                ),
            )

    def stop(self, handle: SubagentHandle, *, reason: str) -> SubagentCancelResult:
        with _admitted_authorized_binding(self, handle) as binding:
            if binding is None:
                return SubagentCancelResult(False, unknown_handle=True)
            result = binding.service.stop(handle, reason=reason)
            if result.already_terminal:
                binding.root.owners.mark_terminal(handle, time.time())
            return dataclasses.replace(
                result,
                audit_metadata=binding.service._record_audit_metadata(
                    handle,
                    operation_task_id=binding.task_id,
                    operation_id=_operation_id(binding),
                ),
            )

    def collect(self, handle: SubagentHandle) -> SubagentCompletion:
        with _admitted_authorized_binding(self, handle) as binding:
            if binding is None:
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
            result = binding.service.collect(handle)
            if result.ready:
                binding.root.owners.mark_terminal(
                    handle,
                    result.result.completed_at
                    if result.result is not None
                    and result.result.completed_at is not None
                    else time.time(),
                )
            elif result.diagnostic == "UNKNOWN_HANDLE":
                binding.root.owners.forget(handle)
            return dataclasses.replace(
                result,
                audit_metadata=binding.service._record_audit_metadata(
                    handle,
                    operation_task_id=binding.task_id,
                    operation_id=_operation_id(binding),
                ),
            )

    def status(self, handle: SubagentHandle) -> SubagentStatus:
        binding = _authorized_binding(self, handle)
        if binding is None:
            return SubagentStatus(
                handle, SubagentState.UNKNOWN, time.time(), "UNKNOWN_HANDLE"
            )
        result = binding.service.status(handle)
        _observe_state(binding, handle, result.state)
        return result

    def wait(
        self, handle: SubagentHandle, *, timeout_seconds: Optional[float] = None
    ) -> SubagentTerminalState:
        binding = _authorized_binding(self, handle)
        if binding is None:
            return SubagentTerminalState(
                handle, SubagentState.UNKNOWN, True, diagnostic="UNKNOWN_HANDLE"
            )
        result = binding.service.wait(handle, timeout_seconds=timeout_seconds)
        if result.completed:
            binding.root.owners.mark_terminal(handle, time.time())
        elif result.state is SubagentState.UNKNOWN:
            binding.root.owners.forget(handle)
        return result

    def cancel(self, handle: SubagentHandle, *, reason: str) -> SubagentCancelResult:
        with _admitted_authorized_binding(self, handle) as binding:
            if binding is None:
                return SubagentCancelResult(False, unknown_handle=True)
            result = binding.service.cancel(handle, reason=reason)
            if result.already_terminal:
                binding.root.owners.mark_terminal(handle, time.time())
            return result

    def result(self, handle: SubagentHandle) -> SubagentResult:
        binding = _authorized_binding(self, handle)
        if binding is None:
            return SubagentResult(
                handle, SubagentState.UNKNOWN, False,
                error_classification="UNKNOWN_HANDLE",
            )
        result = binding.service.result(handle)
        if result.ready:
            binding.root.owners.mark_terminal(
                handle,
                result.completed_at if result.completed_at is not None else time.time(),
            )
        elif result.terminal_state is SubagentState.UNKNOWN:
            binding.root.owners.forget(handle)
        return result

    def reconnect(self, handle: SubagentHandle) -> SubagentReconnectResult:
        binding = _authorized_binding(self, handle)
        if binding is None:
            return SubagentReconnectResult(
                False, SubagentState.UNKNOWN, "RECONNECT_UNAVAILABLE"
            )
        result = binding.service.reconnect(handle)
        _observe_state(binding, handle, result.state)
        return result


_BINDING_LOCK = threading.RLock()
_BINDINGS: weakref.WeakKeyDictionary[BoundSubagentLifecycle, _Binding] = (
    weakref.WeakKeyDictionary()
)


class _ExecutionAuthorityLease:
    """Shared, revocable authority copied with an execution context."""

    __slots__ = ("root", "operation_id", "_active", "_lock")

    def __init__(self, root: _RootAuthority, operation_id: str) -> None:
        self.root = root
        self.operation_id = operation_id
        self._active = True
        self._lock = threading.Lock()

    def authorizes(self, root: _RootAuthority) -> bool:
        with self._lock:
            return self._active and self.root is root

    def invalidate(self) -> None:
        with self._lock:
            self._active = False


_ACTIVE_PLUGIN_AUTHORITY: contextvars.ContextVar[
    Optional[_ExecutionAuthorityLease]
] = (
    contextvars.ContextVar("hermes_active_plugin_authority", default=None)
)


@contextmanager
def _binding_profile_scope(binding: _Binding):
    """Anchor launch config/secrets to the host-minted canonical profile."""
    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_authoritative_secret_scope,
        set_authoritative_secret_scope,
    )
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    home = Path(binding.root.authority.canonical_profile_key)
    home_token = set_hermes_home_override(str(home))
    secret_token = None
    try:
        secret_token = set_authoritative_secret_scope(
            build_profile_secret_scope(home)
        )
        yield
    finally:
        if secret_token is not None:
            reset_authoritative_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _binding_for(facade: BoundSubagentLifecycle) -> Optional[_Binding]:
    with _BINDING_LOCK:
        return _BINDINGS.get(facade)


class _BoundSubagentAuthorityScope:
    """Reusable per-dispatch scope with separately revocable phase leases."""

    __slots__ = ("_facade", "_operation_id", "_lease", "_token")

    def __init__(self, facade: BoundSubagentLifecycle) -> None:
        self._facade = facade
        self._operation_id = secrets.token_urlsafe(18)
        self._lease: Optional[_ExecutionAuthorityLease] = None
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        if self._lease is not None or self._token is not None:
            raise RuntimeError("Plugin authority scope is already active.")
        binding = _binding_for(self._facade)
        if (
            binding is None
            or not binding.root.can_mint()
            or binding.root.owners.is_revoked()
        ):
            raise SubagentLifecycleError("Plugin lifecycle authority is unavailable.")
        lease = _ExecutionAuthorityLease(binding.root, self._operation_id)
        self._lease = lease
        self._token = _ACTIVE_PLUGIN_AUTHORITY.set(lease)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        lease = self._lease
        token = self._token
        self._lease = None
        self._token = None
        if lease is not None:
            # ContextVar values are copied by reference into spawned asyncio
            # tasks. Invalidate each phase lease before restoring the caller,
            # while retaining only the non-authoritative operation ID for the
            # awaited phase of this same dispatch.
            lease.invalidate()
        if token is not None:
            _ACTIVE_PLUGIN_AUTHORITY.reset(token)


class _SuppressedSubagentAuthorityScope:
    """Reusable neutral scope for all phases of one legacy dispatch."""

    __slots__ = ("_token",)

    def __init__(self) -> None:
        self._token: Optional[contextvars.Token] = None

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("Plugin authority suppression is already active.")
        self._token = _ACTIVE_PLUGIN_AUTHORITY.set(None)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        token = self._token
        self._token = None
        if token is not None:
            _ACTIVE_PLUGIN_AUTHORITY.reset(token)


def _bind_bound_subagent_authority(
    facade: BoundSubagentLifecycle,
) -> _BoundSubagentAuthorityScope:
    """Create one host-minted authority scope for a complete dispatch."""
    return _BoundSubagentAuthorityScope(facade)


def _suppress_bound_subagent_authority() -> _SuppressedSubagentAuthorityScope:
    """Run a legacy plugin handler without inheriting an outer authority."""
    return _SuppressedSubagentAuthorityScope()


def _active_authority_matches(binding: _Binding) -> bool:
    lease = _ACTIVE_PLUGIN_AUTHORITY.get()
    return lease is not None and lease.authorizes(binding.root)


def _operation_id(binding: _Binding) -> Optional[str]:
    if binding.operation_id is not None:
        return binding.operation_id
    lease = _ACTIVE_PLUGIN_AUTHORITY.get()
    if lease is not None and lease.authorizes(binding.root):
        return lease.operation_id
    return None


def _mint(binding: _Binding) -> BoundSubagentLifecycle:
    facade = object.__new__(BoundSubagentLifecycle)
    with _BINDING_LOCK:
        if not binding.root.can_mint() or binding.root.owners.is_revoked():
            return facade
        _BINDINGS[facade] = binding
    return facade


def _mint_bound_subagent_lifecycle(
    service: SubagentLifecycleService,
    *,
    plugin_id: str,
    profile_path: Path,
    manager_scope_key: str,
    parent_resolver: Callable[[], Any],
) -> BoundSubagentLifecycle:
    owners = _OwnerStore()
    root = _RootAuthority(
        _PluginAuthority(
            plugin_id=str(plugin_id),
            canonical_profile_key=str(profile_path.expanduser().resolve(strict=False)),
            manager_scope_key=str(manager_scope_key),
        ),
        owners,
    )
    service._bind_record_expiry_observer(owners.forget)
    return _mint(_Binding(
        service=service,
        root=root,
        parent_resolver=parent_resolver,
    ))


def _mint_invocation_facade(
    base: BoundSubagentLifecycle,
    *,
    session_id: Optional[str],
    task_id: Optional[str],
    operation_id: str,
) -> BoundSubagentLifecycle:
    binding = _binding_for(base)
    if binding is None:
        return object.__new__(BoundSubagentLifecycle)
    return _mint(dataclasses.replace(
        binding,
        session_id=str(session_id) if session_id is not None else None,
        task_id=str(task_id) if task_id is not None else None,
        operation_id=operation_id,
    ))


def _revoke_bound_subagent_lifecycle(facade: BoundSubagentLifecycle) -> None:
    binding = _binding_for(facade)
    if binding is None:
        return
    # Close lifecycle admission and wait without holding the identity registry
    # or lifecycle registry locks. Every admitted control finishes, and every
    # admitted launch publishes ownership, before this wait can finish.
    binding.root.begin_revoke_and_wait()
    with _BINDING_LOCK:
        for candidate, candidate_binding in list(_BINDINGS.items()):
            if candidate_binding.root is binding.root:
                _BINDINGS.pop(candidate, None)
        # Revocation shares the identity-registry critical section so a
        # concurrent invocation derivation cannot publish a new binding after
        # every existing facade has been removed.
        binding.root.owners.revoke()
    binding.service._unbind_record_expiry_observer()


def _active_session(binding: _Binding) -> Optional[str]:
    parent = binding.parent_resolver()
    return str(getattr(parent, "session_id", "") or "") or None


def _expected_session(binding: _Binding) -> Optional[str]:
    active_session = _active_session(binding)
    return binding.session_id if binding.session_id is not None else active_session


def _active_binding(
    facade: BoundSubagentLifecycle,
) -> Optional[_Binding]:
    binding = _binding_for(facade)
    if binding is None or not _active_authority_matches(binding):
        return None
    active_session = _active_session(binding)
    if active_session != _expected_session(binding):
        return None
    return binding


def _authorized_binding(
    facade: BoundSubagentLifecycle, handle: SubagentHandle
) -> Optional[_Binding]:
    if not isinstance(handle, SubagentHandle):
        return None
    binding = _active_binding(facade)
    if binding is None:
        return None
    expected_session = _expected_session(binding)
    if not binding.root.owners.authorize(handle, expected_session):
        return None
    return binding


@contextmanager
def _admitted_active_binding(facade: BoundSubagentLifecycle):
    """Keep one active lifecycle operation linearizable with plugin unload."""
    binding = _active_binding(facade)
    if binding is None or not binding.root.admit_operation():
        yield None
        return
    try:
        yield binding
    finally:
        binding.root.complete_operation()


@contextmanager
def _admitted_authorized_binding(
    facade: BoundSubagentLifecycle, handle: SubagentHandle
):
    """Keep one authorized control operation linearizable with plugin unload."""
    if not isinstance(handle, SubagentHandle):
        yield None
        return
    binding = _active_binding(facade)
    if binding is None or not binding.root.admit_operation():
        yield None
        return
    try:
        expected_session = _expected_session(binding)
        if not binding.root.owners.authorize(handle, expected_session):
            yield None
            return
        yield binding
    finally:
        binding.root.complete_operation()


def _observe_state(
    binding: _Binding, handle: SubagentHandle, state: SubagentState
) -> None:
    if state in _TERMINAL_STATES:
        binding.root.owners.mark_terminal(handle, time.time())
    elif state is SubagentState.UNKNOWN:
        binding.root.owners.forget(handle)


@dataclasses.dataclass(frozen=True)
class PluginToolInvocation:
    invocation_contract_version: int
    plugin_id: str
    session_id: Optional[str]
    task_id: Optional[str]
    operation_id: str
    profile_name: str
    execution_kind: str
    delegation_depth: int
    delegation_role: str
    platform: str
    subagents: BoundSubagentLifecycle


def _bounded_public_execution_identifier(value: Any, *, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    normalized = value.strip().lower()
    if (
        not normalized
        or len(normalized) > 64
        or any(
            not character.isascii()
            or not (character.isalnum() or character in {"-", "_"})
            for character in normalized
        )
    ):
        return fallback
    return normalized


def _host_execution_identity(binding: _Binding) -> tuple[str, int, str, str]:
    """Derive bounded invocation context from the active host parent only."""
    from agent.delegation_context import classify_delegation_depth

    parent = binding.parent_resolver()
    if parent is None:
        execution_kind, depth = classify_delegation_depth(None)
        return execution_kind, depth, "unknown", "unknown"

    execution_kind, depth = classify_delegation_depth(
        getattr(parent, "_delegate_depth", None)
    )
    if execution_kind == "delegated":
        role = _bounded_public_execution_identifier(
            getattr(parent, "_delegate_role", "leaf"), fallback="leaf"
        )
        if role not in {"leaf", "orchestrator"}:
            role = "leaf"
    else:
        role = "root" if execution_kind == "root" else "unknown"
    platform = _bounded_public_execution_identifier(
        getattr(parent, "platform", "unknown"), fallback="unknown"
    )
    return execution_kind, depth, role, platform


def _make_plugin_tool_invocation(
    *,
    profile_name: str,
    session_id: Optional[str],
    task_id: Optional[str],
    subagents: BoundSubagentLifecycle,
) -> PluginToolInvocation:
    binding = _binding_for(subagents)
    if binding is None or binding.root.owners.is_revoked():
        raise SubagentLifecycleError("Plugin lifecycle authority is unavailable.")
    lease = _ACTIVE_PLUGIN_AUTHORITY.get()
    if lease is None or not lease.authorizes(binding.root):
        raise SubagentLifecycleError("Plugin lifecycle authority is unavailable.")
    operation_id = lease.operation_id
    execution_kind, delegation_depth, delegation_role, platform = (
        _host_execution_identity(binding)
    )
    return PluginToolInvocation(
        invocation_contract_version=PLUGIN_INVOCATION_CONTRACT_VERSION,
        plugin_id=binding.root.authority.plugin_id,
        session_id=str(session_id) if session_id is not None else None,
        task_id=str(task_id) if task_id is not None else None,
        operation_id=operation_id,
        profile_name=profile_name,
        execution_kind=execution_kind,
        delegation_depth=delegation_depth,
        delegation_role=delegation_role,
        platform=platform,
        subagents=_mint_invocation_facade(
            subagents,
            session_id=session_id,
            task_id=task_id,
            operation_id=operation_id,
        ),
    )
