"""Process-wide admission for Hermes delegation background units.

The admission layer consumes host facts only: pause state, tree depth, batch
cardinality, and the process-wide unit limit.  A single dispatch -- including
a fan-out batch -- occupies one background unit.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Optional


PAUSED = "PAUSED"
DEPTH_REACHED = "DEPTH_REACHED"
CAPACITY_REACHED = "CAPACITY_REACHED"
BATCH_DISABLED = "BATCH_DISABLED"
BATCH_TOO_LARGE = "BATCH_TOO_LARGE"
INVALID_SPAWN_FACTS = "INVALID_SPAWN_FACTS"


class _AdmissionState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.paused = False
        self.active_units = 0
        self.generation = 0


_STATE = _AdmissionState()
_LEASE_MINT = object()


class AdmissionLease:
    """Exactly-once lease for one admitted background unit."""

    __slots__ = ("_generation", "_released", "_release_lock")

    def __init__(self, mint: object, generation: int) -> None:
        if mint is not _LEASE_MINT:
            raise TypeError("AdmissionLease instances are host-minted")
        self._generation = generation
        self._released = False
        self._release_lock = threading.Lock()

    def release(self) -> bool:
        """Release once; return whether this call owned the release."""
        with self._release_lock:
            if self._released:
                return False
            self._released = True
        with _STATE.lock:
            # A test reset invalidates old leases.  They must never decrement
            # capacity acquired by a later generation.
            if self._generation == _STATE.generation:
                _STATE.active_units = max(0, _STATE.active_units - 1)
        return True

    def __enter__(self) -> "AdmissionLease":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


@dataclasses.dataclass(frozen=True)
class AdmissionDecision:
    lease: Optional[AdmissionLease] = None
    rejection_code: Optional[str] = None

    @property
    def admitted(self) -> bool:
        return self.lease is not None


def set_spawn_paused(paused: bool) -> bool:
    with _STATE.lock:
        _STATE.paused = bool(paused)
        return _STATE.paused


def is_spawn_paused() -> bool:
    with _STATE.lock:
        return _STATE.paused


def validate_spawn(
    *,
    parent_depth: int,
    max_spawn_depth: int,
    batch_size: int = 1,
    batch_enabled: bool = True,
    max_batch_size: Optional[int] = None,
) -> Optional[str]:
    """Return a stable rejection code, or ``None`` for valid host facts."""
    with _STATE.lock:
        return _validate_locked(
            enforce_pause=True,
            parent_depth=parent_depth,
            max_spawn_depth=max_spawn_depth,
            batch_size=batch_size,
            batch_enabled=batch_enabled,
            max_batch_size=max_batch_size,
        )


def _validate_locked(
    *,
    enforce_pause: bool,
    parent_depth: Optional[int],
    max_spawn_depth: Optional[int],
    batch_size: int,
    batch_enabled: bool,
    max_batch_size: Optional[int],
) -> Optional[str]:
    if enforce_pause and _STATE.paused:
        return PAUSED
    if (
        parent_depth is not None
        and max_spawn_depth is not None
        and parent_depth >= max_spawn_depth
    ):
        return DEPTH_REACHED
    if batch_size > 1 and not batch_enabled:
        return BATCH_DISABLED
    if max_batch_size is not None and batch_size > max_batch_size:
        return BATCH_TOO_LARGE
    return None


def try_acquire_background_unit(max_background_units: int) -> AdmissionDecision:
    """Atomically reserve one process-wide background unit without queuing."""
    return try_admit_background_unit(max_background_units)


def try_admit_background_unit(
    max_background_units: int,
    *,
    enforce_pause: bool = False,
    parent_depth: Optional[int] = None,
    max_spawn_depth: Optional[int] = None,
    batch_size: int = 1,
    batch_enabled: bool = True,
    max_batch_size: Optional[int] = None,
) -> AdmissionDecision:
    """Validate supplied host facts and reserve capacity in one lock hold."""
    limit = max(1, int(max_background_units))
    with _STATE.lock:
        if enforce_pause and (
            parent_depth is None or max_spawn_depth is None
        ):
            return AdmissionDecision(rejection_code=INVALID_SPAWN_FACTS)
        rejection = _validate_locked(
            enforce_pause=enforce_pause,
            parent_depth=parent_depth,
            max_spawn_depth=max_spawn_depth,
            batch_size=batch_size,
            batch_enabled=batch_enabled,
            max_batch_size=max_batch_size,
        )
        if rejection is not None:
            return AdmissionDecision(rejection_code=rejection)
        if _STATE.active_units >= limit:
            return AdmissionDecision(rejection_code=CAPACITY_REACHED)
        _STATE.active_units += 1
        return AdmissionDecision(
            lease=AdmissionLease(_LEASE_MINT, _STATE.generation)
        )


def active_background_units() -> int:
    with _STATE.lock:
        return _STATE.active_units


def _reset_for_tests() -> None:
    """Invalidate test-created leases and restore process defaults."""
    with _STATE.lock:
        _STATE.generation += 1
        _STATE.active_units = 0
        _STATE.paused = False
