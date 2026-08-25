"""Behavioral contract for process-wide delegation admission."""

import threading

import pytest

from tools import delegation_admission as admission


@pytest.fixture(autouse=True)
def _clean_admission():
    admission._reset_for_tests()
    yield
    admission._reset_for_tests()


def test_pause_depth_and_batch_validation_use_bounded_codes():
    admission.set_spawn_paused(True)
    assert admission.validate_spawn(parent_depth=0, max_spawn_depth=2) == "PAUSED"
    admission.set_spawn_paused(False)
    assert admission.validate_spawn(parent_depth=2, max_spawn_depth=2) == "DEPTH_REACHED"
    assert admission.validate_spawn(
        parent_depth=0, max_spawn_depth=2, batch_size=2, batch_enabled=False
    ) == "BATCH_DISABLED"
    assert admission.validate_spawn(
        parent_depth=0,
        max_spawn_depth=2,
        batch_size=4,
        batch_enabled=True,
        max_batch_size=3,
    ) == "BATCH_TOO_LARGE"
    assert admission.validate_spawn(
        parent_depth=0,
        max_spawn_depth=2,
        batch_size=3,
        batch_enabled=True,
        max_batch_size=3,
    ) is None


def test_legacy_delegate_pause_uses_shared_gate_and_resume_clears_it():
    from tools.delegate_tool import delegate_task

    admission.set_spawn_paused(True)
    paused = delegate_task(goal="do not launch", parent_agent=object())
    assert "spawning is paused" in paused
    admission.set_spawn_paused(False)
    assert admission.is_spawn_paused() is False


def test_pause_and_capacity_decision_is_atomic_under_concurrent_calls():
    """A settled hard pause admits no late racer and consumes no capacity."""
    for _ in range(50):
        admission._reset_for_tests()
        start = threading.Barrier(3)
        decisions = []

        def pause():
            start.wait(timeout=5)
            admission.set_spawn_paused(True)

        def admit():
            start.wait(timeout=5)
            decisions.append(
                admission.try_admit_background_unit(
                    1,
                    enforce_pause=True,
                    parent_depth=0,
                    max_spawn_depth=2,
                )
            )

        threads = [threading.Thread(target=pause), threading.Thread(target=admit)]
        for thread in threads:
            thread.start()
        start.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=5)

        first = decisions[0]
        if first.lease is not None:
            first.lease.release()
        late = admission.try_admit_background_unit(
            1, enforce_pause=True, parent_depth=0, max_spawn_depth=2
        )
        assert late.rejection_code == "PAUSED"
        assert admission.active_background_units() == 0


def test_background_unit_reservation_is_atomic_and_release_is_exactly_once():
    barrier = threading.Barrier(3)
    decisions = []

    def race():
        barrier.wait(timeout=5)
        decisions.append(admission.try_acquire_background_unit(1))

    threads = [threading.Thread(target=race) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert sum(item.admitted for item in decisions) == 1
    assert sorted(item.rejection_code or "" for item in decisions) == [
        "",
        "CAPACITY_REACHED",
    ]
    lease = next(item.lease for item in decisions if item.admitted)
    assert admission.active_background_units() == 1
    assert lease.release() is True
    assert lease.release() is False
    assert admission.active_background_units() == 0


def test_reset_invalidates_old_lease_without_releasing_new_generation():
    old = admission.try_acquire_background_unit(1).lease
    admission._reset_for_tests()
    new = admission.try_acquire_background_unit(1).lease
    assert old.release() is True
    assert admission.active_background_units() == 1
    assert new.release() is True
    assert admission.active_background_units() == 0
