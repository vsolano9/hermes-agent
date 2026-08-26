"""Contract tests for the public plugin subagent lifecycle API."""

import dataclasses
import json
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import agent.subagent_lifecycle as lifecycle_module
from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
    bind_subagent_parent,
    get_active_subagent_parent,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.registry import registry


def _public_route_assessment(**overrides):
    values = {
        "api_contract_version": lifecycle_module.LIFECYCLE_API_CONTRACT_VERSION,
        "route": lifecycle_module.SubagentRouteIdentity("synthetic", "model-a"),
        "eligible": True,
        "reason": "ELIGIBLE",
        "transport": "chat_completions",
        "authenticated": True,
        "agent_capable": True,
        "exact_empty_model_tools": True,
        "mutation_evidence_complete": True,
        "independent_mutation_channels": frozenset(),
        "hermes_model_tool_count": 0,
        "assessed_at": 1.0,
        "assessment_id": "asm_00000000000000000000000000000000",
    }
    values.update(overrides)
    return lifecycle_module.SubagentRouteAssessment(**values)


@pytest.fixture(autouse=True)
def _clean_delegation_admission():
    from tools import delegation_admission

    delegation_admission._reset_for_tests()
    yield
    delegation_admission._reset_for_tests()


@pytest.mark.parametrize(
    "overrides",
    [
        {"eligible": False, "reason": "MUTATION_CHANNEL_UNAVAILABLE"},
        {
            "eligible": False,
            "reason": "MUTATION_CHANNEL_UNAVAILABLE",
            "authenticated": False,
            "independent_mutation_channels": frozenset({"EXTERNAL_PROCESS"}),
        },
        {
            "eligible": False,
            "reason": "MUTATION_CHANNEL_UNAVAILABLE",
            "agent_capable": False,
            "independent_mutation_channels": frozenset({"EXTERNAL_PROCESS"}),
        },
        {
            "eligible": False,
            "reason": "MUTATION_CHANNEL_UNAVAILABLE",
            "transport": "unknown",
        },
        {
            "eligible": False,
            "reason": "MUTATION_CHANNEL_UNAVAILABLE",
            "independent_mutation_channels": frozenset({"UNKNOWN_TRANSPORT"}),
        },
        {
            "eligible": False,
            "reason": "MUTATION_CHANNEL_UNAVAILABLE",
            "exact_empty_model_tools": False,
            "hermes_model_tool_count": 1,
        },
        {
            "eligible": False,
            "reason": "MUTATION_CHANNEL_UNAVAILABLE",
            "exact_empty_model_tools": False,
            "mutation_evidence_complete": False,
            "hermes_model_tool_count": 1,
        },
        {
            "eligible": False,
            "reason": "MUTATION_CHANNEL_UNAVAILABLE",
            "exact_empty_model_tools": False,
            "mutation_evidence_complete": False,
            "independent_mutation_channels": frozenset({"HERMES_MODEL_TOOLS"}),
        },
    ],
)
def test_route_assessment_rejects_semantically_impossible_unavailable_receipts(
    overrides,
):
    with pytest.raises(SubagentLifecycleError, match="Malformed"):
        _public_route_assessment(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "transport": "unknown",
            "independent_mutation_channels": frozenset({"UNKNOWN_TRANSPORT"}),
        },
        {
            "independent_mutation_channels": frozenset({"EXTERNAL_PROCESS"}),
        },
        {
            "exact_empty_model_tools": False,
            "hermes_model_tool_count": 2,
            "independent_mutation_channels": frozenset({"HERMES_MODEL_TOOLS"}),
        },
        {
            "exact_empty_model_tools": False,
            "mutation_evidence_complete": False,
        },
    ],
)
def test_route_assessment_accepts_each_exact_unavailable_blocker(overrides):
    assessment = _public_route_assessment(
        eligible=False,
        reason="MUTATION_CHANNEL_UNAVAILABLE",
        **overrides,
    )

    assert assessment.eligible is False
    assert assessment.reason == "MUTATION_CHANNEL_UNAVAILABLE"


class FakeChild:
    def __init__(self, ident="sa-test"):
        self._subagent_id = ident
        self._delegate_role = "leaf"
        self._delegate_depth = 1
        self.provider = "test"
        self.model = "test-model"
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self.valid_tool_names = set()
        self.tools = []
        self.interrupted = False
        self.interrupt_kind = None
        self.interrupt_message = None
        self.tool_reason = None

    def interrupt(self, _reason):
        self.interrupted = True
        self.interrupt_kind = "soft"

    def hard_interrupt(self, reason, *, tool_reason=None):
        self.interrupted = True
        self.interrupt_kind = "hard"
        self.interrupt_message = reason
        self.tool_reason = tool_reason


@pytest.fixture
def lifecycle(monkeypatch):
    parent = SimpleNamespace(session_id="parent-1", enabled_toolsets=["file"])
    counter = iter(range(1000))

    def build(**_kwargs):
        return FakeChild(f"sa-{next(counter)}")

    def run(_index, _goal, child, _parent):
        for _ in range(20):
            if child.interrupted:
                return {
                    "status": "interrupted",
                    "summary": None,
                    "api_calls": 0,
                    "duration_seconds": 0,
                }
            time.sleep(0.002)
        return {
            "status": "completed",
            "summary": "safe summary",
            "api_calls": 1,
            "duration_seconds": 0.01,
        }

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    return SubagentLifecycleService(lambda: parent)






def test_cancel_is_cooperative_and_forged_handle_is_unknown(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    assert lifecycle.cancel(handle, reason="test").accepted
    terminal = lifecycle.wait(handle, timeout_seconds=1)
    assert terminal.state is SubagentState.CANCELLED
    forged = handle.__class__(**{**handle.to_dict(), "capability": "forged"})
    assert lifecycle.status(forged).state is SubagentState.UNKNOWN
    assert lifecycle.result(forged).error_classification == "UNKNOWN_HANDLE"
    other_parent = SimpleNamespace(session_id="different-parent")
    other_service = SubagentLifecycleService(lambda: other_parent)
    assert other_service.status(handle).state is SubagentState.UNKNOWN
    from tools import delegation_admission
    assert delegation_admission.active_background_units() == 0


def test_capabilities_are_immutable_and_version_api_separately_from_handle(lifecycle):
    capabilities = lifecycle.capabilities()

    assert capabilities.api_contract_version == 3
    assert capabilities.handle_serialization_version == 1
    assert capabilities.features == frozenset(
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
    )
    assert capabilities.providers_are_host_resolved is True
    assert capabilities.working_directory_supported is False
    assert capabilities.restart_recovery == "unsupported"
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        capabilities.api_contract_version = 3


def test_v2_toolset_modes_are_unambiguous_and_v1_empty_still_inherits(monkeypatch):
    parent = SimpleNamespace(
        session_id="toolset-parent",
        enabled_toolsets=["file"],
        provider="test",
        model="test-model",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
    )
    observed = []
    counter = iter(range(20))

    def build(**kwargs):
        observed.append(kwargs["toolsets"])
        return FakeChild(f"sa-toolsets-{next(counter)}")

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed",
            "summary": "ok",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    service = SubagentLifecycleService(lambda: parent)

    for request in (
        SubagentLaunchRequest(goal="v1 none", allowed_toolsets=None),
        SubagentLaunchRequest(goal="v1 empty", allowed_toolsets=()),
        lifecycle_module.SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="v2 inherit"),
            toolset_mode="inherit",
        ),
        lifecycle_module.SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="v2 zero"),
            toolset_mode="exact",
            exact_toolsets=(),
        ),
        lifecycle_module.SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="v2 subset"),
            toolset_mode="exact",
            exact_toolsets=("file",),
        ),
    ):
        handle = service.launch(request)
        assert service.wait(handle, timeout_seconds=1).completed is True

    assert observed == [None, None, None, [], ["file"]]

    invalid = (
        lifecycle_module.SubagentLaunchRequestV2(
            api_contract_version=1,
            base=SubagentLaunchRequest(goal="wrong version"),
        ),
        lifecycle_module.SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="ambiguous", allowed_toolsets=()),
        ),
        lifecycle_module.SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="ambiguous inherit"),
            toolset_mode="inherit",
            exact_toolsets=("file",),
        ),
    )
    for request in invalid:
        with pytest.raises(SubagentLifecycleError):
            service.launch(request)
    assert len(observed) == 5


@pytest.mark.parametrize("role", ["leaf", "orchestrator"])
def test_v2_exact_subset_uses_authoritative_parent_composite_expansion(
    monkeypatch, role
):
    parent = SimpleNamespace(
        session_id=f"composite-{role}", enabled_toolsets=["hermes-cli"]
    )
    observed = []
    counter = iter(range(20))

    def build(**kwargs):
        observed.append((kwargs["toolsets"], kwargs["exact_toolsets"], kwargs["role"]))
        return FakeChild(f"sa-composite-{role}-{next(counter)}")

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed",
            "summary": "ok",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    service = SubagentLifecycleService(lambda: parent)
    accepted = lifecycle_module.SubagentLaunchRequestV2(
        api_contract_version=2,
        base=SubagentLaunchRequest(goal="composite file subset", role=role),
        toolset_mode="exact",
        exact_toolsets=("file",),
    )
    handle = service.launch(accepted)
    assert service.wait(handle, timeout_seconds=1).completed is True
    assert observed == [(["file"], True, role)]

    for rejected_toolset in ("x_search", "delegation"):
        with pytest.raises(SubagentLifecycleError):
            service.launch(
                lifecycle_module.SubagentLaunchRequestV2(
                    api_contract_version=2,
                    base=SubagentLaunchRequest(
                        goal=f"reject {rejected_toolset}", role=role
                    ),
                    toolset_mode="exact",
                    exact_toolsets=(rejected_toolset,),
                )
            )

    raw_parent = SimpleNamespace(
        session_id=f"raw-{role}", enabled_toolsets=["file"]
    )
    raw_service = SubagentLifecycleService(lambda: raw_parent)
    raw_handle = raw_service.launch(accepted)
    assert raw_service.wait(raw_handle, timeout_seconds=1).completed is True
    with pytest.raises(SubagentLifecycleError):
        raw_service.launch(
            lifecycle_module.SubagentLaunchRequestV2(
                api_contract_version=2,
                base=SubagentLaunchRequest(goal="raw parent broadening", role=role),
                toolset_mode="exact",
                exact_toolsets=("terminal",),
            )
        )


def test_list_and_lookup_expose_only_this_lifecycle_direct_children(monkeypatch):
    parent = SimpleNamespace(session_id="direct-parent", enabled_toolsets=["file"])
    counter = iter(range(20))
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: FakeChild(f"sa-direct-{next(counter)}"),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed",
            "summary": "ok",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    owner = SubagentLifecycleService(lambda: parent)
    nested_or_foreign = SubagentLifecycleService(lambda: parent)
    owned = owner.launch(SubagentLaunchRequest(goal="owned direct child"))
    foreign = nested_or_foreign.launch(
        SubagentLaunchRequest(goal="nested child without parent handle")
    )
    assert owner.wait(owned, timeout_seconds=1).completed is True
    assert nested_or_foreign.wait(foreign, timeout_seconds=1).completed is True

    listed = owner.list()
    assert tuple(item.handle for item in listed) == (owned,)
    assert owner.status(foreign).state is SubagentState.UNKNOWN


def test_steer_reports_queued_missed_terminal_unknown_and_unsupported(monkeypatch):
    parent = SimpleNamespace(session_id="steer-parent", enabled_toolsets=["file"])
    gate = threading.Event()
    entered = threading.Event()
    children = []

    class SteerChild(FakeChild):
        def __init__(self, ident, accepted):
            super().__init__(ident)
            self.accepted = accepted
            self.messages = []

        def steer(self, text):
            self.messages.append(text)
            return self.accepted

    def run(*_args):
        entered.set()
        assert gate.wait(timeout=60)
        return {
            "status": "completed",
            "summary": "done",
            "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: children.pop(0),
    )
    service = SubagentLifecycleService(lambda: parent)

    queued_child = SteerChild("sa-steer-queued", True)
    children.append(queued_child)
    queued_handle = service.launch(SubagentLaunchRequest(goal="queued"))
    assert entered.wait(timeout=5)
    queued = service.steer(queued_handle, "change direction")
    assert queued.disposition is lifecycle_module.SubagentControlDisposition.QUEUED
    assert queued.accepted is True
    assert queued_child.messages == ["change direction"]
    assert not hasattr(service, "message")
    gate.set()
    assert service.wait(queued_handle, timeout_seconds=1).completed is True
    assert (
        service.steer(queued_handle, "too late").disposition
        is lifecycle_module.SubagentControlDisposition.TERMINAL
    )

    gate.clear()
    entered.clear()
    missed_child = SteerChild("sa-steer-missed", False)
    children.append(missed_child)
    missed_handle = service.launch(SubagentLaunchRequest(goal="missed"))
    assert entered.wait(timeout=5)
    missed = service.steer(missed_handle, "try once")
    assert missed.disposition is lifecycle_module.SubagentControlDisposition.MISSED
    assert missed.accepted is False
    assert missed_child.messages == ["try once"]
    gate.set()
    service.wait(missed_handle, timeout_seconds=1)

    gate.clear()
    entered.clear()
    children.append(FakeChild("sa-steer-unsupported"))
    unsupported_handle = service.launch(SubagentLaunchRequest(goal="unsupported"))
    assert entered.wait(timeout=5)
    unsupported = service.steer(unsupported_handle, "cannot queue")
    assert (
        unsupported.disposition
        is lifecycle_module.SubagentControlDisposition.UNSUPPORTED
    )
    gate.set()
    service.wait(unsupported_handle, timeout_seconds=1)

    forged = dataclasses.replace(queued_handle, capability="forged")
    unknown = service.steer(forged, "not authorized")
    assert unknown.disposition is lifecycle_module.SubagentControlDisposition.UNKNOWN_HANDLE
    assert unknown.accepted is False


def test_collect_is_not_ready_then_stable_idempotent_and_owner_content_is_bounded(
    monkeypatch,
):
    parent = SimpleNamespace(session_id="collect-parent", enabled_toolsets=["file"])
    gate = threading.Event()
    entered = threading.Event()
    owner_canary = (
        "sk-synthetic https://secret.invalid/v1 ENV_CANARY=/private/owner/path "
    )
    summary = owner_canary + ("x" * 40_000)
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: FakeChild("sa-collect"),
    )

    def run(*_args):
        entered.set()
        assert gate.wait(timeout=60)
        return {
            "status": "completed",
            "summary": summary,
            "structured_payload": {
                "nested": {"items": ["immutable-owner-content"]}
            },
            "api_calls": 1,
            "duration_seconds": 0.25,
        }

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="collect once ready"))
    assert entered.wait(timeout=5)

    pending = service.collect(handle)
    assert pending.api_contract_version == 3
    assert pending.handle_serialization_version == 1
    assert pending.ready is False
    assert pending.terminal_state is None
    assert pending.result is None
    assert pending.event_id is None
    assert pending.collected_at is None
    assert pending.diagnostic is None

    gate.set()
    assert service.wait(handle, timeout_seconds=1).completed is True
    first = service.collect(handle)
    second = service.collect(handle)
    assert first == second
    assert first.ready is True
    assert first.terminal_state is SubagentState.SUCCEEDED
    assert first.result is not None
    assert first.event_id.startswith("subagent-completion-")
    assert first.collected_at is not None
    assert first.result.result_hash
    assert owner_canary in first.result.summary
    assert len(first.result.summary) == lifecycle_module._MAX_RESULT_CHARS
    assert first.result.structured_payload["nested"]["items"] == (
        "immutable-owner-content",
    )
    with pytest.raises(TypeError):
        first.result.usage_metadata["api_calls"] = 99
    with pytest.raises(AttributeError):
        first.result.usage_metadata._items = ()
    with pytest.raises(TypeError):
        first.result.structured_payload["nested"]["added"] = True
    with pytest.raises(AttributeError):
        first.result.structured_payload["nested"]["items"].append("mutated")
    assert service.collect(handle) == first

    forged = dataclasses.replace(handle, capability="forged")
    unknown = service.collect(forged)
    assert unknown.ready is False
    assert unknown.terminal_state is None
    assert unknown.result is None
    assert unknown.event_id is None
    assert unknown.collected_at is None
    assert unknown.diagnostic == "UNKNOWN_HANDLE"
    assert owner_canary not in unknown.diagnostic


def test_collect_result_hash_is_bound_to_immutable_nested_payload(monkeypatch):
    parent = SimpleNamespace(session_id="hash-parent", enabled_toolsets=["file"])
    child_ids = iter(("sa-hash-one", "sa-hash-two"))
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: FakeChild(next(child_ids)),
    )

    def run(_index, goal, *_args):
        payload = (
            {"nested": {"goal": goal, "values": [1, 2]}}
            if goal == "payload-one"
            else {"blob": "x" * (lifecycle_module._MAX_METADATA_BYTES * 2)}
        )
        return {
            "status": "completed",
            "summary": "same summary",
            "structured_payload": payload,
            "api_calls": 1,
            "duration_seconds": 0.1,
        }

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    service = SubagentLifecycleService(lambda: parent)
    first_handle = service.launch(SubagentLaunchRequest(goal="payload-one"))
    second_handle = service.launch(SubagentLaunchRequest(goal="payload-two"))
    assert service.wait(first_handle, timeout_seconds=1).completed is True
    assert service.wait(second_handle, timeout_seconds=1).completed is True

    first = service.collect(first_handle)
    repeated = service.collect(first_handle)
    second = service.collect(second_handle)
    assert first == repeated
    assert first.result.result_hash == repeated.result.result_hash
    assert first.result.result_hash != second.result.result_hash
    assert second.result.structured_payload["truncated"] is True
    assert "blob" not in second.result.structured_payload
    with pytest.raises(TypeError):
        first.result.structured_payload["nested"]["goal"] = "tampered"


def test_stop_is_the_versioned_alias_for_existing_cancel_semantics(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="stop me"))

    stopped = lifecycle.stop(handle, reason="owner requested stop")
    assert stopped.accepted is True
    assert stopped.state is SubagentState.CANCEL_REQUESTED
    assert lifecycle.wait(handle, timeout_seconds=1).state is SubagentState.CANCELLED

    already_terminal = lifecycle.stop(handle, reason="repeat")
    assert already_terminal.accepted is False
    assert already_terminal.already_terminal is True
    assert already_terminal.state is SubagentState.CANCELLED


def test_blocking_steer_does_not_hold_registry_and_terminal_race_is_missed(
    monkeypatch,
):
    parent = SimpleNamespace(session_id="steer-race-parent", enabled_toolsets=["file"])
    steer_entered = threading.Event()
    release_steer = threading.Event()
    finish_run = threading.Event()

    class BlockingChild(FakeChild):
        def steer(self, _text):
            steer_entered.set()
            assert release_steer.wait(timeout=60)
            return True

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: BlockingChild("sa-steer-race"),
    )

    def run(*_args):
        assert finish_run.wait(timeout=60)
        return {
            "status": "completed",
            "summary": "done",
            "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="steer race"))
    steer_result = []
    steer_thread = threading.Thread(
        target=lambda: steer_result.append(service.steer(handle, "race"))
    )
    steer_thread.start()
    assert steer_entered.wait(timeout=5)

    lookup_done = threading.Event()
    lookup_result = []

    def lookup():
        lookup_result.append(service.list())
        lookup_done.set()

    lookup_thread = threading.Thread(target=lookup)
    lookup_thread.start()
    try:
        assert lookup_done.wait(timeout=0.5), "steer callback held the registry lock"
        assert tuple(item.handle for item in lookup_result[0]) == (handle,)
        finish_run.set()
        assert service.wait(handle, timeout_seconds=1).completed is True
    finally:
        release_steer.set()
        finish_run.set()
        steer_thread.join(timeout=2)
        lookup_thread.join(timeout=2)

    assert steer_result[0].disposition is lifecycle_module.SubagentControlDisposition.MISSED
    assert steer_result[0].accepted is False


def test_core_generated_result_metadata_is_allowlisted_numeric_and_bounded(
    monkeypatch,
):
    parent = SimpleNamespace(session_id="metadata-parent", enabled_toolsets=["file"])
    metadata_canaries = (
        "sk-metadata-canary https://metadata.invalid/v1 "
        "ENV_SECRET=canary /private/metadata/path"
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: FakeChild("sa-metadata"),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed",
            "summary": "owner-authorized result",
            "api_calls": metadata_canaries,
            "duration_seconds": {"raw": metadata_canaries},
        },
    )
    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="metadata receipt"))
    assert service.wait(handle, timeout_seconds=1).completed is True

    result = service.result(handle)
    assert result.usage_metadata == {"api_calls": 0}
    assert result.tool_execution_summary == {"duration_seconds": 0.0}
    core_metadata = json.dumps(
        {
            "usage": result.usage_metadata,
            "tools": result.tool_execution_summary,
            "status": dataclasses.asdict(service.status(handle)),
            "completion_diagnostic": service.collect(handle).diagnostic,
        },
        default=str,
        sort_keys=True,
    )
    assert metadata_canaries not in core_metadata


def test_child_status_and_internal_exception_cannot_escape_into_core_diagnostics(
    monkeypatch,
):
    parent = SimpleNamespace(session_id="diagnostic-parent", enabled_toolsets=["file"])
    canary = (
        "sk-diagnostic-canary https://diagnostic.invalid/v1 "
        "ENV_SECRET=canary /private/diagnostic/path"
    )
    child_ids = iter(("sa-status-canary", "sa-exception-canary"))
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: FakeChild(next(child_ids)),
    )

    class CredentialPathCanaryError(RuntimeError):
        pass

    outcomes = iter(
        (
            {
                "status": canary,
                "summary": canary,
                "error": canary,
                "api_calls": 0,
                "duration_seconds": 0,
            },
            CredentialPathCanaryError(canary),
        )
    )

    def run(*_args):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    service = SubagentLifecycleService(lambda: parent)

    status_handle = service.launch(SubagentLaunchRequest(goal="hostile status"))
    assert service.wait(status_handle, timeout_seconds=1).completed is True
    status_result = service.result(status_handle)
    assert status_result.summary == canary
    assert status_result.error_classification == "CHILD_FAILED"
    assert status_result.error_message == "Child reported failure."

    exception_handle = service.launch(
        SubagentLaunchRequest(goal="hostile internal exception")
    )
    assert service.wait(exception_handle, timeout_seconds=1).completed is True
    exception_result = service.result(exception_handle)
    assert exception_result.error_classification == "INTERNAL_ERROR"
    assert exception_result.error_message == "Child execution failed."

    core_diagnostics = json.dumps(
        {
            "status_classification": status_result.error_classification,
            "status_message": status_result.error_message,
            "exception_classification": exception_result.error_classification,
            "exception_message": exception_result.error_message,
        },
        sort_keys=True,
    )
    assert canary not in core_diagnostics
    assert "CredentialPathCanaryError" not in core_diagnostics


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("contract_version", 2),
        ("subagent_id", "sa-forged-id"),
        ("parent_session_id", "forged-parent"),
        ("correlation_id", "forged-correlation"),
        ("created_at", 123.5),
        ("provider", "forged-provider"),
        ("model", "forged-model"),
        ("role", "orchestrator"),
        ("depth", 99),
        ("capability", "forged-capability"),
    ),
)
def test_every_serialized_handle_field_is_authenticated(lifecycle, field, value):
    handle = lifecycle.launch(
        SubagentLaunchRequest(
            goal="authenticate every field", correlation_id=f"owned-{field}"
        )
    )
    assert lifecycle.wait(handle, timeout_seconds=1).completed is True
    forged = dataclasses.replace(handle, **{field: value})

    assert lifecycle.status(forged).state is SubagentState.UNKNOWN
    assert (
        lifecycle.steer(forged, "forged").disposition
        is lifecycle_module.SubagentControlDisposition.UNKNOWN_HANDLE
    )
    assert lifecycle.stop(forged, reason="forged").unknown_handle is True
    assert lifecycle.collect(forged).diagnostic == "UNKNOWN_HANDLE"


def test_actual_nested_descendant_is_not_owned_by_top_level_lifecycle(monkeypatch):
    root_parent = SimpleNamespace(
        session_id="root-session", enabled_toolsets=["file"], _delegate_depth=0
    )
    children = []

    def build(**kwargs):
        child = FakeChild(f"sa-depth-{len(children) + 1}")
        child._delegate_depth = int(getattr(kwargs["parent_agent"], "_delegate_depth", 0)) + 1
        children.append(child)
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr("tools.delegate_tool._get_max_spawn_depth", lambda: 3)
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed",
            "summary": "done",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    root = SubagentLifecycleService(lambda: root_parent)
    direct = root.launch(
        SubagentLaunchRequest(goal="direct orchestrator", role="orchestrator")
    )
    assert root.wait(direct, timeout_seconds=1).completed is True

    direct_child_parent = SimpleNamespace(
        session_id="direct-child-session",
        enabled_toolsets=["file"],
        _delegate_depth=direct.depth,
        _subagent_id=direct.subagent_id,
    )
    nested_service = SubagentLifecycleService(lambda: direct_child_parent)
    grandchild = nested_service.launch(SubagentLaunchRequest(goal="grandchild"))
    assert grandchild.depth == 2
    assert nested_service.wait(grandchild, timeout_seconds=1).completed is True

    assert tuple(item.handle for item in root.list()) == (direct,)
    assert root.status(grandchild).state is SubagentState.UNKNOWN
    assert (
        root.steer(grandchild, "not owned").disposition
        is lifecycle_module.SubagentControlDisposition.UNKNOWN_HANDLE
    )
    assert root.stop(grandchild, reason="not owned").unknown_handle is True
    assert root.collect(grandchild).diagnostic == "UNKNOWN_HANDLE"


def test_process_restart_simulation_has_no_native_recovery(lifecycle, monkeypatch):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="pre-restart child"))
    assert lifecycle.wait(handle, timeout_seconds=1).completed is True

    with lifecycle_module._REGISTRY.lock:
        lifecycle_module._REGISTRY.records.pop(handle.subagent_id, None)
        if handle.correlation_id:
            lifecycle_module._REGISTRY.correlations.pop(
                (handle.parent_session_id, handle.correlation_id), None
            )
    monkeypatch.setattr(lifecycle_module, "_SECRET", b"new-process-generation")
    restarted = SubagentLifecycleService(lifecycle._parent_agent_resolver)

    assert restarted.status(handle).state is SubagentState.UNKNOWN
    reconnect = restarted.reconnect(handle)
    assert reconnect.connected is False
    assert reconnect.diagnostic == "RECONNECT_UNAVAILABLE"
    assert (
        restarted.steer(handle, "after restart").disposition
        is lifecycle_module.SubagentControlDisposition.UNKNOWN_HANDLE
    )
    assert restarted.stop(handle, reason="after restart").unknown_handle is True
    assert restarted.collect(handle).diagnostic == "UNKNOWN_HANDLE"


def test_lifecycle_launch_obeys_shared_pause_and_resume(lifecycle):
    from tools.delegate_tool import set_spawn_paused

    set_spawn_paused(True)
    try:
        with pytest.raises(SubagentLifecycleError, match="PAUSED"):
            lifecycle.launch(SubagentLaunchRequest(goal="must not build"))
    finally:
        set_spawn_paused(False)

    handle = lifecycle.launch(SubagentLaunchRequest(goal="resume works"))
    assert lifecycle.wait(handle, timeout_seconds=1).completed is True


def test_lifecycle_launch_obeys_shared_spawn_depth_before_child_build(monkeypatch):
    parent = SimpleNamespace(
        session_id="parent-at-depth",
        enabled_toolsets=["file"],
        _delegate_depth=2,
    )
    build = Mock()
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr("tools.delegate_tool._get_max_spawn_depth", lambda: 2)
    service = SubagentLifecycleService(lambda: parent)

    with pytest.raises(SubagentLifecycleError, match="DEPTH_REACHED"):
        service.launch(SubagentLaunchRequest(goal="must not build"))

    build.assert_not_called()


def test_lifecycle_capacity_rejects_before_child_construction_and_releases(monkeypatch):
    from tools import delegation_admission as admission

    admission._reset_for_tests()
    parent = SimpleNamespace(
        session_id="parent-capacity", enabled_toolsets=["file"], _delegate_depth=0
    )
    build = Mock()
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr("tools.delegate_tool._get_max_concurrent_children", lambda: 1)
    held = admission.try_acquire_background_unit(1).lease
    service = SubagentLifecycleService(lambda: parent)
    try:
        with pytest.raises(SubagentLifecycleError, match="CAPACITY_REACHED"):
            service.launch(SubagentLaunchRequest(goal="must not build"))
        build.assert_not_called()
    finally:
        held.release()

    child = FakeChild("sa-capacity-release")
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", lambda **_kw: child
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed",
            "summary": "ok",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    handle = service.launch(SubagentLaunchRequest(goal="now admitted"))
    assert service.wait(handle, timeout_seconds=1).completed is True
    assert admission.active_background_units() == 0


def test_duplicate_correlation_is_reserved_before_concurrent_build(monkeypatch):
    parent = SimpleNamespace(
        session_id="parent-correlation", enabled_toolsets=["file"], _delegate_depth=0
    )
    correlation_id = f"corr-{uuid.uuid4().hex}"
    build_entered = threading.Event()
    release_build = threading.Event()
    build_count = 0
    build_lock = threading.Lock()

    def build(**_kwargs):
        nonlocal build_count
        with build_lock:
            build_count += 1
            ident = build_count
        build_entered.set()
        assert release_build.wait(timeout=60)
        return FakeChild(f"sa-correlation-{ident}")

    monkeypatch.setattr("tools.delegate_tool._get_max_concurrent_children", lambda: 2)
    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed", "summary": "ok", "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    service = SubagentLifecycleService(lambda: parent)
    outcomes = []

    def launch():
        try:
            outcomes.append(
                service.launch(
                    SubagentLaunchRequest(goal="race", correlation_id=correlation_id)
                )
            )
        except Exception as exc:
            outcomes.append(exc)

    first = threading.Thread(target=launch)
    second = threading.Thread(target=launch)
    first.start()
    assert build_entered.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release_build.set()
    first.join(timeout=5)

    assert build_count == 1
    assert sum(isinstance(item, SubagentLifecycleError) for item in outcomes) == 1
    assert "Duplicate correlation_id" in str(
        next(item for item in outcomes if isinstance(item, SubagentLifecycleError))
    )
    handle = next(item for item in outcomes if not isinstance(item, Exception))
    assert service.wait(handle, timeout_seconds=1).completed is True


def test_correlation_failure_and_expiry_remove_only_their_exact_owner(
    monkeypatch,
):
    import agent.subagent_lifecycle as lifecycle_module

    parent = SimpleNamespace(
        session_id="parent-correlation-owner",
        enabled_toolsets=["file"],
        _delegate_depth=0,
    )
    service = SubagentLifecycleService(lambda: parent)
    failed_correlation = f"corr-fail-{uuid.uuid4().hex}"
    failed_key = (parent.session_id, failed_correlation)

    def replace_then_fail(**_kwargs):
        with lifecycle_module._REGISTRY.lock:
            lifecycle_module._REGISTRY.correlations[failed_key] = "other-owner"
        raise RuntimeError("build failure")

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        replace_then_fail,
    )
    with pytest.raises(
        SubagentLifecycleError, match="Hermes failed to launch subagent"
    ) as caught:
        service.launch(
            SubagentLaunchRequest(goal="fail", correlation_id=failed_correlation)
        )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    with lifecycle_module._REGISTRY.lock:
        assert lifecycle_module._REGISTRY.correlations[failed_key] == "other-owner"
        lifecycle_module._REGISTRY.correlations.pop(failed_key)

    expiry_correlation = f"corr-expiry-{uuid.uuid4().hex}"
    expiry_key = (parent.session_id, expiry_correlation)
    child = FakeChild("sa-correlation-expiry")
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: child,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed", "summary": "ok", "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    handle = service.launch(
        SubagentLaunchRequest(goal="expire", correlation_id=expiry_correlation)
    )
    assert service.wait(handle, timeout_seconds=1).completed is True
    completed_at = service.result(handle).completed_at
    with lifecycle_module._REGISTRY.lock:
        lifecycle_module._REGISTRY.correlations[expiry_key] = "replacement-owner"
    monkeypatch.setattr(
        "agent.subagent_lifecycle.time.time", lambda: completed_at + 3_601
    )
    assert service.status(handle).state is SubagentState.UNKNOWN
    with lifecycle_module._REGISTRY.lock:
        assert (
            lifecycle_module._REGISTRY.correlations[expiry_key]
            == "replacement-owner"
        )
        lifecycle_module._REGISTRY.correlations.pop(expiry_key)


def test_correlation_reservation_is_released_after_admission_denial(lifecycle):
    from tools.delegate_tool import set_spawn_paused

    correlation_id = f"corr-paused-{uuid.uuid4().hex}"
    set_spawn_paused(True)
    try:
        with pytest.raises(SubagentLifecycleError, match="PAUSED"):
            lifecycle.launch(
                SubagentLaunchRequest(goal="paused", correlation_id=correlation_id)
            )
    finally:
        set_spawn_paused(False)

    handle = lifecycle.launch(
        SubagentLaunchRequest(goal="retry", correlation_id=correlation_id)
    )
    assert lifecycle.wait(handle, timeout_seconds=1).completed is True


def test_lifecycle_unit_saturates_direct_async_dispatch_and_build_failure_releases(
    monkeypatch,
):
    from tools import async_delegation as async_registry
    from tools import delegation_admission as admission

    parent = SimpleNamespace(
        session_id="parent-mixed", enabled_toolsets=["file"], _delegate_depth=0
    )
    gate = threading.Event()
    child = FakeChild("sa-mixed")

    def run_mixed(*_args):
        assert gate.wait(timeout=60)
        return {
            "status": "completed",
            "summary": "ok",
            "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setattr("tools.delegate_tool._get_max_concurrent_children", lambda: 1)
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", lambda **_kw: child
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        run_mixed,
    )
    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="hold shared unit"))
    rejected = async_registry.dispatch_async_delegation(
        goal="direct async",
        context=None,
        toolsets=None,
        role="leaf",
        model="m",
        session_key="",
        runner=lambda: {},
        max_async_children=1,
        enforce_spawn_controls=False,
    )
    assert rejected["rejection_code"] == "CAPACITY_REACHED"
    gate.set()
    assert service.wait(handle, timeout_seconds=1).completed is True
    assert admission.active_background_units() == 0

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("build failed")),
    )
    with pytest.raises(
        SubagentLifecycleError, match="Hermes failed to launch subagent"
    ) as caught:
        service.launch(SubagentLaunchRequest(goal="construction failure"))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert admission.active_background_units() == 0

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kw: FakeChild("sa-submit-failure"),
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle._EXECUTOR",
        SimpleNamespace(
            submit=lambda *_args: (_ for _ in ()).throw(RuntimeError("submit failed"))
        ),
    )
    with pytest.raises(
        SubagentLifecycleError, match="Hermes failed to launch subagent"
    ) as caught:
        service.launch(SubagentLaunchRequest(goal="submit failure"))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert admission.active_background_units() == 0


def test_cancel_uses_explicit_hard_interrupt(lifecycle):
    handle = lifecycle.launch(SubagentLaunchRequest(goal="x"))
    record = lifecycle._record(handle)
    assert record is not None and record.agent is not None

    assert lifecycle.cancel(handle, reason="explicit user cancel").accepted

    assert record.agent.interrupt_kind == "hard"
    assert "explicit user cancel" in record.agent.interrupt_message
    assert record.agent.tool_reason == "subagent cancellation requested"
    lifecycle.wait(handle, timeout_seconds=1)


def test_record_expiry_observer_failure_never_blocks_authoritative_cleanup(
    lifecycle, monkeypatch
):
    observed = []

    def broken_observer(handle):
        observed.append(handle.subagent_id)
        raise RuntimeError("observer failure must be isolated")

    lifecycle._bind_record_expiry_observer(broken_observer)
    expired = lifecycle.launch(SubagentLaunchRequest(goal="expire"))
    assert lifecycle.wait(expired, timeout_seconds=1).completed is True
    completed_at = lifecycle.result(expired).completed_at
    assert completed_at is not None
    monkeypatch.setattr(
        "agent.subagent_lifecycle.time.time", lambda: completed_at + 3_601
    )

    replacement = lifecycle.launch(SubagentLaunchRequest(goal="cleanup continues"))
    assert replacement.subagent_id != expired.subagent_id
    assert observed == [expired.subagent_id]
    assert lifecycle.status(expired).state is SubagentState.UNKNOWN
    assert lifecycle.wait(replacement, timeout_seconds=1).completed is True
    lifecycle._unbind_record_expiry_observer()


def test_first_direct_lookup_cleans_expired_record_and_notifies_owner(
    lifecycle, monkeypatch
):
    expired = []
    lifecycle._bind_record_expiry_observer(
        lambda handle: expired.append(handle.subagent_id)
    )
    handle = lifecycle.launch(SubagentLaunchRequest(goal="direct expiry lookup"))
    assert lifecycle.wait(handle, timeout_seconds=1).completed is True
    completed_at = lifecycle.result(handle).completed_at
    assert completed_at is not None
    monkeypatch.setattr(
        "agent.subagent_lifecycle.time.time", lambda: completed_at + 3_601
    )

    assert lifecycle.status(handle).state is SubagentState.UNKNOWN
    assert expired == [handle.subagent_id]
    assert lifecycle.result(handle).error_classification == "UNKNOWN_HANDLE"








def test_public_lifecycle_runs_host_aggregation(monkeypatch):
    memory = Mock()
    parent = SimpleNamespace(
        session_id="parent-aggregate",
        enabled_toolsets=["file"],
        _memory_manager=memory,
        _current_turn_id="turn-1",
        session_estimated_cost_usd=1.0,
        session_cost_source="none",
        session_cost_status="unknown",
    )
    child = FakeChild("sa-aggregate")
    child.session_id = "child-session"
    hook = Mock()

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **_kwargs: child)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda *_args, **_kwargs: {
            "task_index": 0,
            "status": "completed",
            "summary": "aggregated",
            "api_calls": 1,
            "duration_seconds": 0.25,
            "_child_role": "leaf",
            "_child_cost_usd": 2.5,
        },
    )
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(SubagentLaunchRequest(goal="aggregate me"))
    assert service.wait(handle, timeout_seconds=1).state is SubagentState.SUCCEEDED

    memory.on_delegation.assert_called_once_with(
        task="aggregate me", result="aggregated", child_session_id="child-session"
    )
    hook.assert_called_once_with(
        "subagent_stop",
        parent_session_id="parent-aggregate",
        parent_turn_id="turn-1",
        child_session_id="child-session",
        child_role="leaf",
        child_summary="aggregated",
        child_status="completed",
        # Redacted tool history rides the shared finalization pipeline
        # (#62011/#72403); empty here because the fabricated result carries
        # no tool_trace.
        tool_call_history=[],
        duration_ms=250,
    )
    assert parent.session_estimated_cost_usd == 3.5
    assert parent.session_cost_source == "subagent"
    assert parent.session_cost_status == "estimated"




def test_agent_turn_binds_and_clears_lifecycle_parent(monkeypatch):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    observed = []

    def run_conversation(parent, *_args, **_kwargs):
        observed.append(get_active_subagent_parent())
        return {"final_response": "ok"}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", run_conversation)

    assert agent.run_conversation("hello") == {"final_response": "ok"}
    assert observed == [agent]
    assert get_active_subagent_parent() is None


def test_bound_lifecycle_rejects_plugin_and_manager_scope_replay(
    tmp_path, monkeypatch
):
    parent = SimpleNamespace(session_id="bound-parent", enabled_toolsets=["file"])
    child = FakeChild("sa-bound")
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: child,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "ok", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    def bound(plugin_id, scope):
        manager = PluginManager(scope_key=scope)
        return PluginContext(
            PluginManifest(name=plugin_id, key=plugin_id, source="user"),
            manager,
        )

    owner = bound("owner", str(tmp_path / "scope-a"))
    wrong_plugin = bound("other", str(tmp_path / "scope-a"))
    wrong_scope = bound("owner", str(tmp_path / "scope-b"))
    owner_facades = []
    handles = []

    def schema(name):
        return {
            "name": name,
            "description": "lifecycle authority probe",
            "parameters": {"type": "object", "properties": {}},
        }

    def owner_handler(_args, *, invocation):
        owner_facades.append(invocation.subagents)
        handle = invocation.subagents.launch(SubagentLaunchRequest(goal="bounded"))
        handles.append(handle)
        return invocation.subagents.wait(handle, timeout_seconds=1).state.value

    def replay_handler(_args, *, invocation):
        return owner_facades[0].status(handles[0]).state.value

    registrations = [
        owner.register_tool(
            "_bound_owner", "debugging", schema("_bound_owner"), owner_handler,
        ),
        wrong_plugin.register_tool(
            "_bound_wrong_plugin", "debugging",
            schema("_bound_wrong_plugin"), replay_handler,
        ),
        wrong_scope.register_tool(
            "_bound_wrong_scope", "debugging",
            schema("_bound_wrong_scope"), replay_handler,
        ),
    ]
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_bound_owner", {}, scope=str((tmp_path / "scope-a").resolve()),
                session_id="bound-parent", task_id="turn-owner",
            ) == SubagentState.SUCCEEDED.value
            assert registry.dispatch(
                "_bound_wrong_plugin", {},
                scope=str((tmp_path / "scope-a").resolve()),
                session_id="bound-parent", task_id="turn-replay",
            ) == SubagentState.UNKNOWN.value
            assert registry.dispatch(
                "_bound_wrong_scope", {},
                scope=str((tmp_path / "scope-b").resolve()),
                session_id="bound-parent", task_id="turn-replay",
            ) == SubagentState.UNKNOWN.value
    finally:
        for registration in registrations:
            assert registration is not None
            registration.dispose()
