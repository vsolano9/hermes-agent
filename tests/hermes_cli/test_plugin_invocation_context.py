"""Behavior contracts for opt-in plugin tool invocation context."""

import asyncio
import copy
import dataclasses
import inspect
import itertools
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import agent.subagent_lifecycle as lifecycle_module
import hermes_cli.plugin_invocation as plugin_invocation_module
from agent.subagent_lifecycle import (
    SubagentHandle,
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentState,
    bind_subagent_parent,
)
from hermes_cli.plugin_invocation import BoundSubagentLifecycle, PluginToolInvocation
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.registry import registry


def _context(tmp_path, plugin_id="invocation-probe", *, scope_name="profile-a"):
    home = tmp_path / scope_name
    home.mkdir(exist_ok=True)
    manager = PluginManager(scope_key=str(home.resolve()))
    manifest = PluginManifest(name=plugin_id, key=plugin_id, source="user")
    return PluginContext(manifest, manager), manager


def _schema(name):
    return {
        "name": name,
        "description": "invocation probe",
        "parameters": {"type": "object", "properties": {}},
    }


def test_explicit_invocation_parameter_receives_dispatch_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    context, manager = _context(tmp_path)
    expected_profile_name = context.profile_name
    seen = []

    def handler(args, *, invocation, **kwargs):
        seen.append((args, invocation, kwargs))
        return "ok"

    registration = context.register_tool(
        "_invocation_identity_probe",
        "debugging",
        _schema("_invocation_identity_probe"),
        handler,
    )
    try:
        result = registry.dispatch(
            "_invocation_identity_probe",
            {"value": 1},
            scope=manager.scope_key,
            session_id="session-1",
            task_id="turn-a",
            user_task="original task",
            existing="kept",
        )
        assert result == "ok"
        args, invocation, kwargs = seen[0]
        assert args == {"value": 1}
        assert isinstance(invocation, PluginToolInvocation)
        assert invocation.invocation_contract_version == 2
        assert invocation.plugin_id == "invocation-probe"
        assert invocation.session_id == "session-1"
        assert invocation.task_id == "turn-a"
        assert isinstance(invocation.operation_id, str)
        assert 16 <= len(invocation.operation_id) <= 64
        assert invocation.operation_id not in {"session-1", "turn-a"}
        assert invocation.profile_name == expected_profile_name
        assert not hasattr(invocation, "parent_agent")
        assert manager.scope_key not in repr(invocation)
        assert "canonical_profile_key" not in repr(invocation)
        with pytest.raises(dataclasses.FrozenInstanceError):
            invocation.task_id = "tampered"
        assert not hasattr(invocation.subagents, "__dict__")
        assert not hasattr(invocation.subagents, "_service")
        assert not hasattr(invocation.subagents, "_parent_resolver")
        assert not hasattr(invocation.subagents, "_authority")
        assert not hasattr(invocation.subagents, "parent_agent")
        assert kwargs == {
            "session_id": "session-1",
            "task_id": "turn-a",
            "user_task": "original task",
            "existing": "kept",
        }
    finally:
        assert registration is not None
        registration.dispose()


def test_invocation_execution_identity_is_host_derived_and_sanitized(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "execution-identity")
    seen = []

    def handler(_args, *, invocation):
        seen.append(invocation)
        return "ok"

    registration = context.register_tool(
        "_execution_identity_probe",
        "debugging",
        _schema("_execution_identity_probe"),
        handler,
    )
    parent = SimpleNamespace(
        session_id="identity-session",
        _delegate_depth=2,
        _delegate_role="orchestrator",
        platform="telegram",
    )
    monkeypatch.setenv("HERMES_DELEGATION_DEPTH", "999")
    monkeypatch.setenv("HERMES_PLATFORM", "secret-env-platform")
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_execution_identity_probe",
                {},
                scope=manager.scope_key,
                session_id="identity-session",
                task_id="identity-turn",
            ) == "ok"
        invocation = seen[0]
        assert invocation.invocation_contract_version == 2
        assert invocation.execution_kind == "delegated"
        assert invocation.delegation_depth == 2
        assert invocation.delegation_role == "orchestrator"
        assert invocation.platform == "telegram"
        assert "secret-env-platform" not in repr(invocation)
    finally:
        assert registration is not None
        registration.dispose()


@pytest.mark.parametrize("raw_depth", [None, [], -1, True])
def test_malformed_host_depth_mints_unknown_non_root_invocation(
    tmp_path, raw_depth
):
    context, manager = _context(tmp_path, f"unknown-depth-{type(raw_depth).__name__}")
    seen = []

    registration = context.register_tool(
        "_unknown_execution_identity_probe",
        "debugging",
        _schema("_unknown_execution_identity_probe"),
        lambda _args, *, invocation: seen.append(invocation) or "ok",
    )
    parent = SimpleNamespace(
        session_id="unknown-session",
        _delegate_depth=raw_depth,
        _delegate_role="root",
        platform="cli",
    )
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_unknown_execution_identity_probe",
                {},
                scope=manager.scope_key,
                session_id="unknown-session",
                task_id="unknown-turn",
            ) == "ok"
        assert seen[0].execution_kind == "unknown"
        assert seen[0].delegation_depth == -1
        assert seen[0].delegation_role == "unknown"
    finally:
        registration.dispose()


@pytest.mark.parametrize("raw_depth", [65, 10**100])
def test_every_positive_host_depth_remains_delegated(tmp_path, raw_depth):
    context, manager = _context(tmp_path, f"positive-depth-{raw_depth}")
    seen = []
    registration = context.register_tool(
        "_positive_execution_identity_probe",
        "debugging",
        _schema("_positive_execution_identity_probe"),
        lambda _args, *, invocation: seen.append(invocation) or "ok",
    )
    parent = SimpleNamespace(
        session_id="positive-session",
        _delegate_depth=raw_depth,
        _delegate_role="orchestrator",
        platform="cli",
    )
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_positive_execution_identity_probe",
                {},
                scope=manager.scope_key,
                session_id="positive-session",
                task_id="positive-turn",
            ) == "ok"
        assert seen[0].execution_kind == "delegated"
        assert seen[0].delegation_depth == raw_depth
        assert seen[0].delegation_role == "orchestrator"
    finally:
        registration.dispose()


def test_missing_host_depth_mints_unknown_non_root_invocation(tmp_path):
    context, manager = _context(tmp_path, "missing-depth")
    seen = []
    registration = context.register_tool(
        "_missing_execution_identity_probe",
        "debugging",
        _schema("_missing_execution_identity_probe"),
        lambda _args, *, invocation: seen.append(invocation) or "ok",
    )
    parent = SimpleNamespace(session_id="missing-session", platform="cli")
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_missing_execution_identity_probe",
                {},
                scope=manager.scope_key,
                session_id="missing-session",
                task_id="missing-turn",
            ) == "ok"
        assert seen[0].execution_kind == "unknown"
        assert seen[0].delegation_depth == -1
        assert seen[0].delegation_role == "unknown"
    finally:
        registration.dispose()


def test_bound_route_reads_require_active_authority_and_revoke_with_unload(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "route-authority")
    retained = []

    def handler(_args, *, invocation):
        retained.append(invocation.subagents)
        catalog = invocation.subagents.catalog_routes()
        return json.dumps({"complete": catalog.complete})

    registration = context.register_tool(
        "_route_authority_probe",
        "debugging",
        _schema("_route_authority_probe"),
        handler,
    )
    parent = SimpleNamespace(
        session_id="route-session", provider="synthetic", model="model-a",
        _delegate_depth=0, platform="cli",
    )
    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.catalog_routes",
        lambda self: lifecycle_module.SubagentRouteCatalog(
            api_contract_version=3,
            complete=True,
            routes=(),
            candidate_count=0,
            reason="COMPLETE",
            assessed_at=1.0,
            snapshot_id="snap_00000000000000000000000000000000",
        ),
    )
    try:
        with bind_subagent_parent(parent):
            assert json.loads(registry.dispatch(
                "_route_authority_probe", {}, scope=manager.scope_key,
                session_id="route-session", task_id="route-turn",
            )) == {"complete": True}
        with pytest.raises(SubagentLifecycleError):
            retained[0].catalog_routes()
        plugin_invocation_module._revoke_bound_subagent_lifecycle(
            context.subagent_lifecycle
        )
        with bind_subagent_parent(parent), pytest.raises(SubagentLifecycleError):
            retained[0].catalog_routes()
    finally:
        if registration is not None and registration.active:
            registration.dispose()


def test_route_catalog_admission_drains_before_unload_revokes(tmp_path, monkeypatch):
    context, manager = _context(tmp_path, "route-unload-race")
    entered = threading.Event()
    release = threading.Event()
    retained = []
    outcomes = []

    def catalog(_service):
        entered.set()
        assert release.wait(timeout=5)
        return lifecycle_module.SubagentRouteCatalog(
            api_contract_version=3,
            complete=True,
            routes=(),
            candidate_count=0,
            reason="COMPLETE",
            assessed_at=1.0,
            snapshot_id="snap_00000000000000000000000000000000",
        )

    def handler(_args, *, invocation):
        retained.append(invocation.subagents)
        return invocation.subagents.catalog_routes().reason

    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.catalog_routes", catalog
    )
    registration = context.register_tool(
        "_route_unload_race",
        "debugging",
        _schema("_route_unload_race"),
        handler,
    )
    parent = SimpleNamespace(session_id="route-race", _delegate_depth=0)

    def dispatch():
        with bind_subagent_parent(parent):
            outcomes.append(
                registry.dispatch(
                    "_route_unload_race",
                    {},
                    scope=manager.scope_key,
                    session_id="route-race",
                    task_id="route-race-turn",
                )
            )

    dispatch_thread = threading.Thread(target=dispatch)
    revoke_thread = threading.Thread(
        target=plugin_invocation_module._revoke_bound_subagent_lifecycle,
        args=(context.subagent_lifecycle,),
    )
    try:
        dispatch_thread.start()
        assert entered.wait(timeout=5)
        revoke_thread.start()
        revoke_thread.join(timeout=0.05)
        assert revoke_thread.is_alive()
        release.set()
        dispatch_thread.join(timeout=5)
        revoke_thread.join(timeout=5)
        assert outcomes == ["COMPLETE"]
        assert not dispatch_thread.is_alive()
        assert not revoke_thread.is_alive()
        with bind_subagent_parent(parent), pytest.raises(SubagentLifecycleError):
            retained[0].catalog_routes()
    finally:
        release.set()
        dispatch_thread.join(timeout=5)
        revoke_thread.join(timeout=5)
        if registration is not None and registration.active:
            registration.dispose()


def test_concurrent_bound_catalogs_use_only_each_authoritative_profile(
    tmp_path, monkeypatch
):
    context_a, manager_a = _context(
        tmp_path, "catalog-profile-a", scope_name="catalog-a"
    )
    context_b, manager_b = _context(
        tmp_path, "catalog-profile-b", scope_name="catalog-b"
    )
    (tmp_path / "catalog-a" / ".env").write_text(
        "OPENAI_API_KEY=profile-a-secret\n", encoding="utf-8"
    )
    (tmp_path / "catalog-b" / ".env").write_text(
        "ANTHROPIC_API_KEY=profile-b-secret\n", encoding="utf-8"
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "process-poison-secret")
    descriptors = [
        SimpleNamespace(
            slug="openai-api", auth_type="api_key",
            api_key_env_vars=("OPENAI_API_KEY",), keyless=False,
        ),
        SimpleNamespace(
            slug="anthropic", auth_type="api_key",
            api_key_env_vars=("ANTHROPIC_API_KEY",), keyless=False,
        ),
        SimpleNamespace(
            slug="openrouter", auth_type="api_key",
            api_key_env_vars=("OPENROUTER_API_KEY",), keyless=False,
        ),
    ]
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog", lambda: descriptors
    )
    monkeypatch.setattr(
        "hermes_cli.models._PROVIDER_MODELS",
        {"openai-api": ["gpt-profile-a"], "anthropic": ["claude-profile-b"]},
    )
    monkeypatch.setattr(
        "hermes_cli.models.OPENROUTER_MODELS", [("poison-model", "")]
    )
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            with_overrides=lambda **_kwargs: SimpleNamespace(
                user_providers={}, custom_providers=[], excluded_providers=[]
            )
        ),
    )

    def register(context, name):
        return context.register_tool(
            name,
            "debugging",
            _schema(name),
            lambda _args, *, invocation: json.dumps([
                [route.provider, route.model]
                for route in invocation.subagents.catalog_routes().routes
            ]),
        )

    registration_a = register(context_a, "_profile_a_catalog")
    registration_b = register(context_b, "_profile_b_catalog")
    parent_a = SimpleNamespace(
        session_id="catalog-a", provider="openai-api", model="gpt-profile-a",
        _delegate_depth=0,
    )
    parent_b = SimpleNamespace(
        session_id="catalog-b", provider="anthropic", model="claude-profile-b",
        _delegate_depth=0,
    )

    def dispatch(name, manager, parent, session):
        with bind_subagent_parent(parent):
            return registry.dispatch(
                name, {}, scope=manager.scope_key, session_id=session,
                task_id=f"{session}-turn",
            )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(
                dispatch, "_profile_a_catalog", manager_a, parent_a, "catalog-a"
            )
            future_b = executor.submit(
                dispatch, "_profile_b_catalog", manager_b, parent_b, "catalog-b"
            )
            assert json.loads(future_a.result(timeout=5)) == [
                ["openai-api", "gpt-profile-a"]
            ]
            assert json.loads(future_b.result(timeout=5)) == [
                ["anthropic", "claude-profile-b"]
            ]
    finally:
        registration_a.dispose()
        registration_b.dispose()


def test_root_scoped_plugin_tool_is_hidden_and_denied_for_child(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "root-scope")
    monkeypatch.setattr("tools.registry.hermes_home_key", lambda: manager.scope_key)
    calls = []

    def handler(_args, *, invocation):
        calls.append(invocation.execution_kind)
        return "root-ok"

    registration = context.register_tool(
        "_root_scope_probe",
        "debugging",
        _schema("_root_scope_probe"),
        handler,
        execution_scope="root",
    )
    root = SimpleNamespace(session_id="root-session", _delegate_depth=0)
    child = SimpleNamespace(session_id="child-session", _delegate_depth=1)
    try:
        with bind_subagent_parent(root):
            assert registry.get_definitions(
                {"_root_scope_probe"}, quiet=True
            )[0]["function"]["name"] == "_root_scope_probe"
            assert registry.dispatch(
                "_root_scope_probe",
                {},
                scope=manager.scope_key,
                session_id="root-session",
                task_id="root-turn",
            ) == "root-ok"
        with bind_subagent_parent(child):
            assert registry.get_definitions({"_root_scope_probe"}, quiet=True) == []
            denied = registry.dispatch(
                "_root_scope_probe",
                {},
                scope=manager.scope_key,
                session_id="child-session",
                task_id="child-turn",
            )
        manual = registry.dispatch(
            "_root_scope_probe",
            {},
            scope=manager.scope_key,
            session_id="root-session",
            task_id="stale-turn",
        )
        assert "root execution context" in denied
        assert "root execution context" in manual
        assert calls == ["root"]
    finally:
        registration.dispose()
    assert registry.get_entry("_root_scope_probe", scope=manager.scope_key) is None


def test_kwargs_alone_remains_exact_legacy_behavior(tmp_path):
    context, manager = _context(tmp_path)
    calls = []

    def handler(args, **kwargs):
        calls.append((args, kwargs))
        return "legacy"

    registration = context.register_tool(
        "_legacy_kwargs_probe",
        "debugging",
        _schema("_legacy_kwargs_probe"),
        handler,
    )
    try:
        assert registry.dispatch(
            "_legacy_kwargs_probe",
            {"value": 2},
            scope=manager.scope_key,
            session_id="session-1",
            task_id="turn-a",
        ) == "legacy"
        assert calls == [(
            {"value": 2},
            {"session_id": "session-1", "task_id": "turn-a"},
        )]
    finally:
        assert registration is not None
        registration.dispose()


def test_legacy_handler_capturing_context_cannot_activate_subagent_authority(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "legacy-authority-probe")
    parent = SimpleNamespace(session_id="legacy-session", enabled_toolsets=["file"])
    service_reached = []

    def build(**_kwargs):
        service_reached.append(True)
        return SimpleNamespace(
            _subagent_id="sa-legacy-authority",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        )

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)

    def handler(_args, **_kwargs):
        try:
            context.subagent_lifecycle.launch(SubagentLaunchRequest(goal="denied"))
        except Exception as exc:
            return type(exc).__name__
        return "LAUNCHED"

    registration = context.register_tool(
        "_legacy_authority_probe",
        "debugging",
        _schema("_legacy_authority_probe"),
        handler,
    )
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_legacy_authority_probe", {}, scope=manager.scope_key,
                session_id="legacy-session", task_id="legacy-turn",
            ) == "SubagentLifecycleError"
        assert service_reached == []
    finally:
        assert registration is not None
        registration.dispose()


@pytest.mark.parametrize("handler_shape", ("sync", "async", "sync-awaitable"))
def test_nested_legacy_handler_suppresses_outer_same_plugin_authority(
    tmp_path, monkeypatch, handler_shape
):
    context, manager = _context(tmp_path, f"nested-legacy-{handler_shape}")
    parent = SimpleNamespace(session_id="nested-legacy-session", enabled_toolsets=["file"])
    child_ids = iter((f"sa-owner-{handler_shape}", f"sa-legacy-{handler_shape}"))
    owned = []

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: SimpleNamespace(
            _subagent_id=next(child_ids),
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        ),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    def legacy_attempt():
        try:
            handle = context.subagent_lifecycle.launch(
                SubagentLaunchRequest(goal="legacy must be denied")
            )
        except Exception as exc:
            return type(exc).__name__
        context.subagent_lifecycle.wait(handle, timeout_seconds=1)
        return "LAUNCHED"

    if handler_shape == "sync":
        def legacy_handler(_args, **_kwargs):
            return legacy_attempt()
    elif handler_shape == "async":
        async def legacy_handler(_args, **_kwargs):
            await asyncio.sleep(0)
            return legacy_attempt()
    else:
        async def legacy_awaitable():
            await asyncio.sleep(0)
            return legacy_attempt()

        def legacy_handler(_args, **_kwargs):
            return legacy_awaitable()

    def outer_handler(_args, *, invocation):
        handle = invocation.subagents.launch(SubagentLaunchRequest(goal="owner"))
        owned.append(handle)
        invocation.subagents.wait(handle, timeout_seconds=1)
        nested = registry.dispatch(
            f"_nested_legacy_{handler_shape}", {}, scope=manager.scope_key,
            session_id="nested-legacy-session", task_id="turn-inner",
        )
        restored = invocation.subagents.status(handle).state.value
        return json.dumps({"nested": nested, "restored": restored})

    registrations = [
        context.register_tool(
            f"_nested_outer_{handler_shape}", "debugging",
            _schema(f"_nested_outer_{handler_shape}"), outer_handler,
        ),
        context.register_tool(
            f"_nested_legacy_{handler_shape}", "debugging",
            _schema(f"_nested_legacy_{handler_shape}"), legacy_handler,
            is_async=handler_shape != "sync",
        ),
    ]
    try:
        with bind_subagent_parent(parent):
            result = json.loads(registry.dispatch(
                f"_nested_outer_{handler_shape}", {}, scope=manager.scope_key,
                session_id="nested-legacy-session", task_id="turn-outer",
            ))
        assert result == {
            "nested": "SubagentLifecycleError",
            "restored": SubagentState.SUCCEEDED.value,
        }
        assert len(owned) == 1
    finally:
        for registration in registrations:
            assert registration is not None
            registration.dispose()


@pytest.mark.parametrize("outcome", ("exception", "cancel"))
def test_nested_legacy_suppression_restores_outer_after_awaitable_failure(
    tmp_path, monkeypatch, outcome
):
    context, manager = _context(tmp_path, f"nested-legacy-{outcome}")
    parent = SimpleNamespace(session_id="nested-failure-session", enabled_toolsets=["file"])
    handles = []

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: SimpleNamespace(
            _subagent_id=f"sa-nested-{outcome}",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        ),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    async def legacy_handler(_args, **_kwargs):
        await asyncio.sleep(0)
        if outcome == "cancel":
            raise asyncio.CancelledError()
        raise RuntimeError("expected legacy failure")

    def outer_handler(_args, *, invocation):
        if not handles:
            handle = invocation.subagents.launch(SubagentLaunchRequest(goal="owner"))
            handles.append(handle)
            invocation.subagents.wait(handle, timeout_seconds=1)
        try:
            nested = registry.dispatch(
                f"_nested_failure_{outcome}", {}, scope=manager.scope_key,
                session_id="nested-failure-session", task_id="turn-inner",
            )
        except asyncio.CancelledError:
            nested = "cancelled"
        return json.dumps({
            "nested": nested,
            "restored": invocation.subagents.status(handles[0]).state.value,
        })

    registrations = [
        context.register_tool(
            f"_nested_failure_outer_{outcome}", "debugging",
            _schema(f"_nested_failure_outer_{outcome}"), outer_handler,
        ),
        context.register_tool(
            f"_nested_failure_{outcome}", "debugging",
            _schema(f"_nested_failure_{outcome}"), legacy_handler, is_async=True,
        ),
    ]
    try:
        with bind_subagent_parent(parent):
            result = json.loads(registry.dispatch(
                f"_nested_failure_outer_{outcome}", {}, scope=manager.scope_key,
                session_id="nested-failure-session", task_id="turn-outer",
            ))
        assert result["restored"] == SubagentState.SUCCEEDED.value
        if outcome == "cancel":
            assert result["nested"] == "cancelled"
        else:
            assert "expected legacy failure" in result["nested"]
    finally:
        for registration in registrations:
            assert registration is not None
            registration.dispose()


def test_legacy_authority_boundary_does_not_mint_lifecycle_facade(tmp_path):
    context, manager = _context(tmp_path, "legacy-no-mint")

    def legacy_handler(_args, **_kwargs):
        return "legacy"

    assert context._subagent_lifecycle is None
    registration = context.register_tool(
        "_legacy_no_mint", "debugging", _schema("_legacy_no_mint"), legacy_handler,
    )
    try:
        assert context._subagent_lifecycle is None
        assert registry.dispatch(
            "_legacy_no_mint", {}, scope=manager.scope_key,
            session_id="legacy-session", task_id="legacy-turn",
        ) == "legacy"
        assert context._subagent_lifecycle is None
    finally:
        assert registration is not None
        registration.dispose()


def test_explicit_registration_flag_requires_keyword_compatible_handler(tmp_path):
    context, manager = _context(tmp_path)

    def incompatible(args):
        return "never"

    with pytest.raises(TypeError, match="invocation"):
        context.register_tool(
            "_invalid_invocation_probe",
            "debugging",
            _schema("_invalid_invocation_probe"),
            incompatible,
            inject_invocation=True,
        )
    assert registry.snapshot_registration(
        "_invalid_invocation_probe", scope=manager.scope_key
    ) is None

    def misplaced(invocation, args, **kwargs):
        return "never"

    with pytest.raises(TypeError, match="invocation"):
        context.register_tool(
            "_positional_invocation_probe",
            "debugging",
            _schema("_positional_invocation_probe"),
            misplaced,
            inject_invocation=True,
        )
    assert registry.snapshot_registration(
        "_positional_invocation_probe", scope=manager.scope_key
    ) is None


def test_auto_opt_in_is_keyword_only_and_legacy_invocation_payload_is_unchanged(
    tmp_path,
):
    context, manager = _context(tmp_path)
    calls = []

    def handler(invocation, **kwargs):
        calls.append((invocation, kwargs))
        return "legacy"

    registration = context.register_tool(
        "_legacy_invocation_name_probe",
        "debugging",
        _schema("_legacy_invocation_name_probe"),
        handler,
    )
    try:
        assert registry.dispatch(
            "_legacy_invocation_name_probe",
            {"payload": "unchanged"},
            scope=manager.scope_key,
            session_id="session",
            task_id="turn",
        ) == "legacy"
        assert calls == [(
            {"payload": "unchanged"},
            {"session_id": "session", "task_id": "turn"},
        )]
        signature = inspect.signature(context.register_tool)
        assert signature.parameters["inject_invocation"].kind is inspect.Parameter.KEYWORD_ONLY
    finally:
        assert registration is not None
        registration.dispose()


def test_keyword_only_handler_filters_historical_dispatch_metadata(tmp_path):
    context, manager = _context(tmp_path)
    calls = []

    def handler(args, *, invocation):
        calls.append((args, invocation.session_id, invocation.task_id))
        return "filtered"

    registration = context.register_tool(
        "_filtered_metadata_probe",
        "debugging",
        _schema("_filtered_metadata_probe"),
        handler,
    )
    try:
        assert registry.dispatch(
            "_filtered_metadata_probe",
            {"payload": 1},
            scope=manager.scope_key,
            session_id="session-filtered",
            task_id="turn-filtered",
            user_task="must not leak as an unexpected keyword",
        ) == "filtered"
        assert calls == [({"payload": 1}, "session-filtered", "turn-filtered")]
    finally:
        assert registration is not None
        registration.dispose()


def test_explicit_registration_flag_opts_kwargs_handler_in(tmp_path):
    context, manager = _context(tmp_path)
    calls = []

    def handler(args, **kwargs):
        calls.append(kwargs)
        return "flagged"

    registration = context.register_tool(
        "_flagged_invocation_probe",
        "debugging",
        _schema("_flagged_invocation_probe"),
        handler,
        inject_invocation=True,
    )
    try:
        assert registry.dispatch(
            "_flagged_invocation_probe",
            {},
            scope=manager.scope_key,
            session_id="session-2",
            task_id="turn-b",
        ) == "flagged"
        assert isinstance(calls[0].pop("invocation"), PluginToolInvocation)
        assert calls == [{"session_id": "session-2", "task_id": "turn-b"}]
    finally:
        assert registration is not None
        registration.dispose()


def test_each_opted_in_execution_gets_a_fresh_host_operation_id(tmp_path):
    context, manager = _context(tmp_path)
    invocations = []

    def handler(args, *, invocation):
        invocations.append(invocation)
        return "ok"

    registration = context.register_tool(
        "_operation_identity_probe",
        "debugging",
        _schema("_operation_identity_probe"),
        handler,
    )
    try:
        for external_id in ("message-a", "message-b"):
            assert registry.dispatch(
                "_operation_identity_probe",
                {},
                scope=manager.scope_key,
                session_id="stable-session",
                task_id="stable-session",
                tool_call_id=external_id,
            ) == "ok"
        assert [item.session_id for item in invocations] == [
            "stable-session", "stable-session"
        ]
        assert [item.task_id for item in invocations] == [
            "stable-session", "stable-session"
        ]
        operation_ids = [item.operation_id for item in invocations]
        assert len(set(operation_ids)) == 2
        assert all(16 <= len(item) <= 64 for item in operation_ids)
        assert not set(operation_ids) & {"message-a", "message-b", "stable-session"}
    finally:
        assert registration is not None
        registration.dispose()


def test_same_session_changed_task_can_control_but_other_authority_cannot(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    owner, owner_manager = _context(tmp_path, "owner")
    other_plugin, other_plugin_manager = _context(tmp_path, "other")
    other_profile, other_profile_manager = _context(
        tmp_path, "owner", scope_name="profile-b"
    )
    child = SimpleNamespace(
        _subagent_id="sa-owned",
        _delegate_role="leaf",
        _delegate_depth=1,
        provider="test",
        model="test-model",
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: child,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "summary": "done",
            "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    handle = None
    observations = []
    replay_facades = []

    def handler(args, *, invocation, **_kwargs):
        nonlocal handle
        if handle is None:
            handle = invocation.subagents.launch(SubagentLaunchRequest(goal="read"))
        else:
            observations.append(invocation.subagents.status(handle).state)
        return "ok"

    registration = owner.register_tool(
        "_authority_probe",
        "debugging",
        _schema("_authority_probe"),
        handler,
    )

    def capture_replay_facade(_args, *, invocation):
        replay_facades.append(invocation.subagents)
        return "captured"

    replay_registrations = [
        other_plugin.register_tool(
            "_other_plugin_replay_probe",
            "debugging",
            _schema("_other_plugin_replay_probe"),
            capture_replay_facade,
        ),
        other_profile.register_tool(
            "_other_profile_replay_probe",
            "debugging",
            _schema("_other_profile_replay_probe"),
            capture_replay_facade,
        ),
    ]
    parent = SimpleNamespace(session_id="session-owner", enabled_toolsets=["file"])
    try:
        with bind_subagent_parent(parent):
            registry.dispatch(
                "_authority_probe", {}, scope=owner_manager.scope_key,
                session_id="session-owner", task_id="turn-a",
            )
            registry.dispatch(
                "_authority_probe", {}, scope=owner_manager.scope_key,
                session_id="session-owner", task_id="turn-b",
            )
            assert observations[-1] is not SubagentState.UNKNOWN
            assert other_plugin.subagent_lifecycle.status(handle).state is SubagentState.UNKNOWN
            assert other_profile.subagent_lifecycle.status(handle).state is SubagentState.UNKNOWN
            registry.dispatch(
                "_other_plugin_replay_probe", {},
                scope=other_plugin_manager.scope_key,
                session_id="session-owner", task_id="turn-replay",
            )
            registry.dispatch(
                "_other_profile_replay_probe", {},
                scope=other_profile_manager.scope_key,
                session_id="session-owner", task_id="turn-replay",
            )
            assert len(replay_facades) == 2
            assert all(
                facade.status(handle).state is SubagentState.UNKNOWN
                for facade in replay_facades
            )

            with pytest.raises(TypeError):
                BoundSubagentLifecycle(
                    object(),
                    plugin_id="owner",
                    profile_path=tmp_path / "profile-a",
                    manager_scope_key=owner_manager.scope_key,
                    parent_resolver=lambda: parent,
                )
            reconstructed = object.__new__(BoundSubagentLifecycle)
            assert reconstructed.status(handle).state is SubagentState.UNKNOWN
            try:
                copied = copy.copy(owner.subagent_lifecycle)
            except TypeError:
                copied = None
            if copied is not None and copied is not owner.subagent_lifecycle:
                assert copied.status(handle).state is SubagentState.UNKNOWN
        with bind_subagent_parent(
            SimpleNamespace(session_id="session-other", enabled_toolsets=["file"])
        ):
            assert owner.subagent_lifecycle.status(handle).state is SubagentState.UNKNOWN
    finally:
        assert registration is not None
        registration.dispose()
        for replay_registration in replay_registrations:
            assert replay_registration is not None
            replay_registration.dispose()


def test_v2_operations_are_turn_stable_direct_child_scoped_and_fail_closed(
    tmp_path, monkeypatch
):
    owner, owner_manager = _context(tmp_path, "v2-owner")
    other, other_manager = _context(tmp_path, "v2-other")
    parent = SimpleNamespace(session_id="v2-session", enabled_toolsets=["file"])
    children = []
    child_ids = iter(("sa-v2-owner", "sa-v2-foreign"))
    handles = {}
    turn_b_receipts = []
    wrong_receipts = []
    operation_ids = {}

    class Child:
        def __init__(self, ident):
            self._subagent_id = ident
            self._delegate_role = "leaf"
            self._delegate_depth = 1
            self.provider = "test"
            self.model = "test-model"
            self.interrupted = False
            self.messages = []

        def steer(self, text):
            self.messages.append(text)
            return True

        def hard_interrupt(self, _reason, *, tool_reason=None):
            self.interrupted = True
            return True

    def build(**_kwargs):
        child = Child(next(child_ids))
        children.append(child)
        return child

    def run(_index, _goal, child, _parent):
        deadline = time.time() + 5
        while time.time() < deadline and not child.interrupted:
            time.sleep(0.002)
        return {
            "status": "interrupted" if child.interrupted else "completed",
            "summary": "done",
            "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)

    def owner_handler(args, *, invocation):
        operation_ids[args["action"]] = invocation.operation_id
        if args["action"] == "launch":
            handles["owner"] = invocation.subagents.launch(
                SubagentLaunchRequest(goal="owner direct child")
            )
            return "launched"
        capabilities = invocation.subagents.capabilities()
        listed = invocation.subagents.list()
        queued = invocation.subagents.steer(handles["owner"], "turn-b direction")
        pending = invocation.subagents.collect(handles["owner"])
        foreign_steer = invocation.subagents.steer(
            handles["foreign"], "must not cross authority"
        )
        foreign_collect = invocation.subagents.collect(handles["foreign"])
        stopped = invocation.subagents.stop(handles["owner"], reason="turn-b stop")
        terminal = invocation.subagents.wait(handles["owner"], timeout_seconds=1)
        legacy_status = invocation.subagents.status(handles["owner"])
        legacy_cancel = invocation.subagents.cancel(
            handles["owner"], reason="legacy compatibility check"
        )
        first = invocation.subagents.collect(handles["owner"])
        second = invocation.subagents.collect(handles["owner"])
        turn_b_receipts.append(
            (
                capabilities,
                listed,
                queued,
                pending,
                foreign_steer,
                foreign_collect,
                stopped,
                terminal,
                legacy_status,
                legacy_cancel,
                first,
                second,
            )
        )
        return "controlled"

    def other_handler(args, *, invocation):
        if args["action"] == "launch":
            handles["foreign"] = invocation.subagents.launch(
                SubagentLaunchRequest(goal="foreign or nested child")
            )
            return "launched"
        wrong_receipts.append(
            (
                invocation.subagents.list(),
                invocation.subagents.steer(handles["owner"], "wrong authority"),
                invocation.subagents.collect(handles["owner"]),
            )
        )
        return "denied"

    registrations = [
        owner.register_tool(
            "_v2_owner_operations", "debugging", _schema("_v2_owner_operations"),
            owner_handler,
        ),
        other.register_tool(
            "_v2_other_operations", "debugging", _schema("_v2_other_operations"),
            other_handler,
        ),
    ]
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_v2_owner_operations", {"action": "launch"},
                scope=owner_manager.scope_key, session_id="v2-session",
                task_id="turn-a",
            ) == "launched"
            assert registry.dispatch(
                "_v2_other_operations", {"action": "launch"},
                scope=other_manager.scope_key, session_id="v2-session",
                task_id="turn-a-other",
            ) == "launched"
            assert registry.dispatch(
                "_v2_other_operations", {"action": "replay"},
                scope=other_manager.scope_key, session_id="v2-session",
                task_id="turn-wrong",
            ) == "denied"
            assert registry.dispatch(
                "_v2_owner_operations", {"action": "control"},
                scope=owner_manager.scope_key, session_id="v2-session",
                task_id="turn-b-changed",
            ) == "controlled"
    finally:
        for registration in registrations:
            assert registration is not None
            registration.dispose()

    capabilities, listed, queued, pending, foreign_steer, foreign_collect, stopped, terminal, legacy_status, legacy_cancel, first, second = turn_b_receipts[0]
    assert capabilities.api_contract_version == 3
    assert tuple(status.handle for status in listed) == (handles["owner"],)
    assert listed[0].audit_metadata.launch_task_id == "turn-a"
    assert listed[0].audit_metadata.operation_task_id == "turn-b-changed"
    assert listed[0].audit_metadata.launch_operation_id == operation_ids["launch"]
    assert listed[0].audit_metadata.operation_id == operation_ids["control"]
    assert operation_ids["launch"] != operation_ids["control"]
    assert queued.disposition is lifecycle_module.SubagentControlDisposition.QUEUED
    assert queued.audit_metadata.launch_task_id == "turn-a"
    assert queued.audit_metadata.operation_task_id == "turn-b-changed"
    assert children[0].messages == ["turn-b direction"]
    assert pending.ready is False
    assert pending.audit_metadata.launch_task_id == "turn-a"
    assert pending.audit_metadata.operation_task_id == "turn-b-changed"
    assert foreign_steer.disposition is lifecycle_module.SubagentControlDisposition.WRONG_AUTHORITY
    assert foreign_collect.ready is False
    assert foreign_collect.diagnostic == "UNKNOWN_HANDLE"
    assert stopped.accepted is True
    assert stopped.audit_metadata.launch_task_id == "turn-a"
    assert stopped.audit_metadata.operation_task_id == "turn-b-changed"
    assert terminal.state is SubagentState.CANCELLED
    assert legacy_status.audit_metadata is None
    assert legacy_cancel.audit_metadata is None
    assert first == second and first.ready is True
    assert first.audit_metadata.launch_task_id == "turn-a"
    assert first.audit_metadata.operation_task_id == "turn-b-changed"
    assert first.audit_metadata.launch_operation_id == operation_ids["launch"]
    assert first.audit_metadata.operation_id == operation_ids["control"]
    wrong_list, wrong_steer, wrong_collect = wrong_receipts[0]
    assert tuple(status.handle for status in wrong_list) == (handles["foreign"],)
    assert wrong_steer.disposition is lifecycle_module.SubagentControlDisposition.WRONG_AUTHORITY
    assert wrong_collect.ready is False


def test_active_handler_authority_rejects_retained_cross_profile_facade(
    tmp_path, monkeypatch
):
    owner, owner_manager = _context(tmp_path, "authority-owner")
    other_profile, other_profile_manager = _context(
        tmp_path, "authority-owner", scope_name="profile-b"
    )
    other_scope, other_scope_manager = _context(
        tmp_path, "authority-owner", scope_name="manager-c"
    )
    parent = SimpleNamespace(session_id="authority-session", enabled_toolsets=["file"])
    retained = []
    handle = []

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: SimpleNamespace(
            _subagent_id="sa-active-authority",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        ),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    def owner_handler(_args, *, invocation):
        retained.append(invocation.subagents)
        if not handle:
            handle.append(invocation.subagents.launch(
                SubagentLaunchRequest(goal="owned")
            ))
            invocation.subagents.wait(handle[0], timeout_seconds=1)
        return invocation.subagents.status(handle[0]).state.value

    def replay_handler(_args, *, invocation):
        assert invocation.plugin_id == "authority-owner"
        return retained[0].status(handle[0]).state.value

    registrations = [
        owner.register_tool(
            "_active_authority_owner", "debugging",
            _schema("_active_authority_owner"), owner_handler,
        ),
        other_profile.register_tool(
            "_active_authority_profile_b", "debugging",
            _schema("_active_authority_profile_b"), replay_handler,
        ),
        other_scope.register_tool(
            "_active_authority_manager_c", "debugging",
            _schema("_active_authority_manager_c"), replay_handler,
        ),
    ]
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_active_authority_owner", {}, scope=owner_manager.scope_key,
                session_id="authority-session", task_id="turn-a",
            ) == SubagentState.SUCCEEDED.value
            assert registry.dispatch(
                "_active_authority_owner", {}, scope=owner_manager.scope_key,
                session_id="authority-session", task_id="turn-b",
            ) == SubagentState.SUCCEEDED.value

            def forbidden_service_access(*_args, **_kwargs):
                raise AssertionError("cross-authority facade reached lifecycle service")

            monkeypatch.setattr(
                "agent.subagent_lifecycle.SubagentLifecycleService.status",
                forbidden_service_access,
            )
            assert registry.dispatch(
                "_active_authority_profile_b", {},
                scope=other_profile_manager.scope_key,
                session_id="authority-session", task_id="turn-replay",
            ) == SubagentState.UNKNOWN.value
            assert registry.dispatch(
                "_active_authority_manager_c", {},
                scope=other_scope_manager.scope_key,
                session_id="authority-session", task_id="turn-replay",
            ) == SubagentState.UNKNOWN.value
    finally:
        for registration in registrations:
            assert registration is not None
            registration.dispose()


@pytest.mark.parametrize("handler_shape", ("async", "sync-awaitable"))
def test_active_authority_spans_complete_awaitable_handler_lifetime(
    tmp_path, monkeypatch, handler_shape
):
    context, manager = _context(tmp_path, f"awaitable-{handler_shape}")
    parent = SimpleNamespace(session_id="awaitable-session", enabled_toolsets=["file"])
    child_count = itertools.count()
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: SimpleNamespace(
            _subagent_id=f"sa-{handler_shape}-{next(child_count)}",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        ),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    async def execute(invocation):
        await asyncio.sleep(0)
        ctx_handle = context.subagent_lifecycle.launch(
            SubagentLaunchRequest(goal="awaited through context")
        )
        invocation_handle = invocation.subagents.launch(
            SubagentLaunchRequest(goal="awaited through invocation")
        )
        terminals = [
            context.subagent_lifecycle.wait(ctx_handle, timeout_seconds=1),
            invocation.subagents.wait(invocation_handle, timeout_seconds=1),
        ]
        ctx_completion = context.subagent_lifecycle.collect(ctx_handle)
        invocation_completion = invocation.subagents.collect(invocation_handle)
        audit = [ctx_completion.audit_metadata, invocation_completion.audit_metadata]
        return json.dumps({
            "states": [item.state.value for item in terminals],
            "operation_id": invocation.operation_id,
            "launch_operation_ids": [item.launch_operation_id for item in audit],
            "control_operation_ids": [item.operation_id for item in audit],
        })

    if handler_shape == "async":
        async def handler(_args, *, invocation):
            return await execute(invocation)
    else:
        def handler(_args, *, invocation):
            return execute(invocation)

    registration = context.register_tool(
        f"_awaitable_authority_{handler_shape}",
        "debugging",
        _schema(f"_awaitable_authority_{handler_shape}"),
        handler,
        is_async=True,
    )
    try:
        with bind_subagent_parent(parent):
            result = json.loads(registry.dispatch(
                f"_awaitable_authority_{handler_shape}", {},
                scope=manager.scope_key,
                session_id="awaitable-session", task_id="turn-await",
            ))
            assert result["states"] == [
                SubagentState.SUCCEEDED.value, SubagentState.SUCCEEDED.value
            ]
            assert result["launch_operation_ids"] == [
                result["operation_id"], result["operation_id"]
            ]
            assert result["control_operation_ids"] == [
                result["operation_id"], result["operation_id"]
            ]
    finally:
        assert registration is not None
        registration.dispose()


def test_authority_scope_restores_after_nested_success_exception_and_cancellation(
    tmp_path, monkeypatch
):
    outer, manager = _context(tmp_path, "scope-outer")
    inner, _ = _context(tmp_path, "scope-inner")
    parent = SimpleNamespace(session_id="scope-session", enabled_toolsets=["file"])
    retained = []
    handles = []

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: SimpleNamespace(
            _subagent_id="sa-scope-restore",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        ),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    async def inner_handler(args, *, invocation):
        await asyncio.sleep(0)
        assert retained[0].status(handles[0]).state is SubagentState.UNKNOWN
        if args["outcome"] == "exception":
            raise RuntimeError("nested failure")
        if args["outcome"] == "cancel":
            raise asyncio.CancelledError()
        return "nested-success"

    def outer_handler(args, *, invocation):
        retained.append(invocation.subagents)
        if args["action"] == "seed":
            handle = invocation.subagents.launch(SubagentLaunchRequest(goal="seed"))
            handles.append(handle)
            return invocation.subagents.wait(handle, timeout_seconds=1).state.value
        if args["action"] == "nested":
            nested = registry.dispatch(
                "_scope_inner", {"outcome": args["outcome"]},
                scope=manager.scope_key, session_id="scope-session",
                task_id="turn-inner",
            )
            restored = invocation.subagents.status(handles[0]).state.value
            return json.dumps({"nested": nested, "restored": restored})
        return invocation.subagents.status(handles[0]).state.value

    registrations = [
        outer.register_tool(
            "_scope_outer", "debugging", _schema("_scope_outer"), outer_handler,
        ),
        inner.register_tool(
            "_scope_inner", "debugging", _schema("_scope_inner"), inner_handler,
            is_async=True,
        ),
    ]
    try:
        with bind_subagent_parent(parent):
            assert registry.dispatch(
                "_scope_outer", {"action": "seed"}, scope=manager.scope_key,
                session_id="scope-session", task_id="turn-seed",
            ) == SubagentState.SUCCEEDED.value
            for outcome in ("success", "exception"):
                result = json.loads(registry.dispatch(
                    "_scope_outer", {"action": "nested", "outcome": outcome},
                    scope=manager.scope_key, session_id="scope-session",
                    task_id=f"turn-{outcome}",
                ))
                assert result["restored"] == SubagentState.SUCCEEDED.value
                if outcome == "success":
                    assert result["nested"] == "nested-success"
                else:
                    assert "nested failure" in result["nested"]

            with pytest.raises(asyncio.CancelledError):
                registry.dispatch(
                    "_scope_outer", {"action": "nested", "outcome": "cancel"},
                    scope=manager.scope_key, session_id="scope-session",
                    task_id="turn-cancel",
                )
            assert retained[0].status(handles[0]).state is SubagentState.UNKNOWN
            assert registry.dispatch(
                "_scope_outer", {"action": "check"}, scope=manager.scope_key,
                session_id="scope-session", task_id="turn-after-cancel",
            ) == SubagentState.SUCCEEDED.value
    finally:
        for registration in registrations:
            assert registration is not None
            registration.dispose()


@pytest.mark.parametrize("outcome", ("return", "exception", "cancellation"))
def test_spawned_child_context_loses_authority_when_handler_execution_ends(
    tmp_path, monkeypatch, outcome
):
    context, manager = _context(tmp_path, f"escaped-context-{outcome}")
    parent = SimpleNamespace(session_id="escaped-context-session", enabled_toolsets=["file"])
    service_calls = []

    def build(**_kwargs):
        service_calls.append(True)
        return SimpleNamespace(
            _subagent_id=f"sa-escaped-{outcome}",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        )

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    async def scenario():
        release_child = asyncio.Event()
        escaped_tasks = []

        async def escaped_child(facade):
            await release_child.wait()
            try:
                handle = facade.launch(SubagentLaunchRequest(goal="escaped child"))
            except Exception as exc:
                return type(exc).__name__
            facade.wait(handle, timeout_seconds=1)
            return "LAUNCHED"

        def handler(_args, *, invocation):
            escaped_tasks.append(asyncio.create_task(
                escaped_child(invocation.subagents)
            ))
            if outcome == "exception":
                raise RuntimeError("handler ended with exception")
            if outcome == "cancellation":
                raise asyncio.CancelledError()
            return "handler-returned"

        registration = context.register_tool(
            f"_escaped_context_{outcome}", "debugging",
            _schema(f"_escaped_context_{outcome}"), handler,
        )
        try:
            with bind_subagent_parent(parent):
                if outcome == "cancellation":
                    with pytest.raises(asyncio.CancelledError):
                        registry.dispatch(
                            f"_escaped_context_{outcome}", {},
                            scope=manager.scope_key,
                            session_id="escaped-context-session", task_id="turn-a",
                        )
                else:
                    dispatch_result = registry.dispatch(
                        f"_escaped_context_{outcome}", {}, scope=manager.scope_key,
                        session_id="escaped-context-session", task_id="turn-a",
                    )
                    if outcome == "return":
                        assert dispatch_result == "handler-returned"
                    else:
                        assert "handler ended with exception" in dispatch_result
            release_child.set()
            assert await escaped_tasks[0] == "SubagentLifecycleError"
        finally:
            assert registration is not None
            registration.dispose()

    asyncio.run(scenario())
    assert service_calls == []


def test_owner_metadata_expires_and_plugin_unload_revokes_facade(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "expiry-owner")
    parent = SimpleNamespace(session_id="expiry-session", enabled_toolsets=["file"])
    child = SimpleNamespace(
        _subagent_id="sa-expiry",
        _delegate_role="leaf",
        _delegate_depth=1,
        provider="test",
        model="test-model",
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: child,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )
    clock = [100.0]
    monkeypatch.setattr("hermes_cli.plugin_invocation.time.time", lambda: clock[0])
    handles = []

    def handler(args, *, invocation):
        if args["action"] == "launch":
            handle = invocation.subagents.launch(
                SubagentLaunchRequest(goal=args["goal"])
            )
            handles.append(handle)
            invocation.subagents.wait(handle, timeout_seconds=1)
            return "ready" if invocation.subagents.result(handle).ready else "not-ready"
        return invocation.subagents.status(handles[0]).state.value

    registration = context.register_tool(
        "_owner_expiry_probe", "debugging", _schema("_owner_expiry_probe"), handler,
    )
    with bind_subagent_parent(parent):
        assert registry.dispatch(
            "_owner_expiry_probe", {"action": "launch", "goal": "expire"},
            scope=manager.scope_key, session_id="expiry-session", task_id="turn-a",
        ) == "ready"
        clock[0] += 3_601
        assert registry.dispatch(
            "_owner_expiry_probe", {"action": "status"},
            scope=manager.scope_key, session_id="expiry-session", task_id="turn-b",
        ) == SubagentState.UNKNOWN.value
    assert registration is not None
    registration.dispose()


def test_unpolled_terminal_owner_expires_with_authoritative_lifecycle_record(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "unpolled-owner")
    parent = SimpleNamespace(session_id="unpolled-session", enabled_toolsets=["file"])
    finished = threading.Event()

    def build(**_kwargs):
        return SimpleNamespace(
            _subagent_id="sa-unpolled",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        )

    def run(*_args, **_kwargs):
        finished.set()
        return {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    clock = [100.0]
    monkeypatch.setattr("agent.subagent_lifecycle.time.time", lambda: clock[0])

    handles = []

    def handler(args, *, invocation):
        if args["action"] == "launch":
            handles.append(invocation.subagents.launch(
                SubagentLaunchRequest(goal="finish unobserved")
            ))
            return "launched"
        return invocation.subagents.status(handles[0]).state.value

    registration = context.register_tool(
        "_unpolled_expiry_probe", "debugging",
        _schema("_unpolled_expiry_probe"), handler,
    )
    with bind_subagent_parent(parent):
        assert registry.dispatch(
            "_unpolled_expiry_probe", {"action": "launch"},
            scope=manager.scope_key, session_id="unpolled-session", task_id="turn-a",
        ) == "launched"
        assert finished.wait(timeout=1)
        clock[0] += 3_601
        assert registry.dispatch(
            "_unpolled_expiry_probe", {"action": "status"},
            scope=manager.scope_key, session_id="unpolled-session", task_id="turn-b",
        ) == SubagentState.UNKNOWN.value

        def stale_authority_reached_service(*_args, **_kwargs):
            raise AssertionError("expired owner metadata authorized a service call")

        monkeypatch.setattr(
            "agent.subagent_lifecycle.SubagentLifecycleService.status",
            stale_authority_reached_service,
        )
        assert registry.dispatch(
            "_unpolled_expiry_probe", {"action": "status"},
            scope=manager.scope_key, session_id="unpolled-session", task_id="turn-c",
        ) == SubagentState.UNKNOWN.value
    assert registration is not None
    registration.dispose()


def test_unload_revokes_every_retained_invocation_derived_facade(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "derived-owner")
    parent = SimpleNamespace(session_id="derived-session", enabled_toolsets=["file"])
    derived = []
    handler_active = threading.Event()
    resume_after_unload = threading.Event()
    outcomes = []

    def handler(_args, *, invocation):
        derived.append(invocation.subagents)
        handler_active.set()
        assert resume_after_unload.wait(timeout=5)
        retained_handle = SubagentHandle(
            contract_version=1,
            subagent_id="sa-derived-unload",
            parent_session_id="derived-session",
            correlation_id="corr-derived-unload",
            created_at=1.0,
            provider="test",
            model="test-model",
            role="leaf",
            depth=1,
            capability="native",
        )
        status = invocation.subagents.status(retained_handle)
        try:
            invocation.subagents.launch(SubagentLaunchRequest(goal="after unload"))
        except Exception as exc:
            launch_error = type(exc).__name__
        else:
            launch_error = "LAUNCHED"
        try:
            invocation.subagents.capabilities()
        except Exception as exc:
            capabilities_error = type(exc).__name__
        else:
            capabilities_error = "AVAILABLE"
        outcomes.append(
            (
                status.state,
                launch_error,
                capabilities_error,
                invocation.subagents.list(),
                invocation.subagents.steer(retained_handle, "after unload").disposition,
                invocation.subagents.stop(
                    retained_handle, reason="after unload"
                ).unknown_handle,
                invocation.subagents.collect(retained_handle).diagnostic,
            )
        )
        return "checked"

    registration = context.register_tool(
        "_derived_unload_probe",
        "debugging",
        _schema("_derived_unload_probe"),
        handler,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: SimpleNamespace(
            _subagent_id="sa-derived-unload",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        ),
    )
    def revoked_authority_reached_service(*_args, **_kwargs):
        raise AssertionError("revoked derived facade reached lifecycle service")

    monkeypatch.setattr(
        "agent.subagent_lifecycle.SubagentLifecycleService.status",
        revoked_authority_reached_service,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        revoked_authority_reached_service,
    )

    def dispatch_in_scope():
        with bind_subagent_parent(parent):
            return registry.dispatch(
                "_derived_unload_probe", {}, scope=manager.scope_key,
                session_id="derived-session", task_id="turn-active",
            )

    with ThreadPoolExecutor(max_workers=1) as executor:
        dispatch_future = executor.submit(dispatch_in_scope)
        assert handler_active.wait(timeout=2)
        assert manager.unload("derived-owner") is True
        resume_after_unload.set()
        assert dispatch_future.result(timeout=5) == "checked"
    assert outcomes == [
        (
            SubagentState.UNKNOWN,
            "SubagentLifecycleError",
            "SubagentLifecycleError",
            (),
            lifecycle_module.SubagentControlDisposition.WRONG_AUTHORITY,
            True,
            "UNKNOWN_HANDLE",
        )
    ]
    assert len(derived) == 1
    assert registration is not None


def test_concurrent_authoritative_cleanup_and_unload_are_idempotent(
    tmp_path, monkeypatch
):
    from agent.subagent_lifecycle import SubagentLifecycleService

    context, manager = _context(tmp_path, "cleanup-race-owner")
    parent = SimpleNamespace(session_id="cleanup-race-session", enabled_toolsets=["file"])
    finished = threading.Event()
    child_ids = iter(("sa-cleanup-race-owned", "sa-cleanup-race-trigger"))

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: SimpleNamespace(
            _subagent_id=next(child_ids),
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        ),
    )

    def run(*_args, **_kwargs):
        finished.set()
        return {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)
    clock = [100.0]
    monkeypatch.setattr("agent.subagent_lifecycle.time.time", lambda: clock[0])
    launched = []

    def handler(_args, *, invocation):
        launched.append(invocation.subagents.launch(
            SubagentLaunchRequest(goal="unpolled race")
        ))
        return "launched"

    registration = context.register_tool(
        "_cleanup_race_probe", "debugging", _schema("_cleanup_race_probe"), handler,
    )
    with bind_subagent_parent(parent):
        assert registry.dispatch(
            "_cleanup_race_probe", {}, scope=manager.scope_key,
            session_id="cleanup-race-session", task_id="turn-a",
        ) == "launched"
    assert finished.wait(timeout=1)
    clock[0] += 3_601
    cleanup_service = SubagentLifecycleService(lambda: parent)
    barrier = threading.Barrier(3)

    def cleanup():
        barrier.wait()
        return cleanup_service.launch(SubagentLaunchRequest(goal="trigger cleanup"))

    def unload():
        barrier.wait()
        return manager.unload("cleanup-race-owner"), manager.unload(
            "cleanup-race-owner"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        cleanup_future = executor.submit(cleanup)
        unload_future = executor.submit(unload)
        barrier.wait()
        cleanup_handle = cleanup_future.result(timeout=2)
        unload_results = unload_future.result(timeout=2)

    assert cleanup_handle.subagent_id == "sa-cleanup-race-trigger"
    assert unload_results == (True, False)
    assert len(launched) == 1
    assert registration is not None


def test_launch_admitted_before_unload_finishes_ownership_before_revocation(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "launch-unload-owner")
    parent = SimpleNamespace(session_id="launch-unload-session", enabled_toolsets=["file"])
    service_entered = threading.Event()
    release_service = threading.Event()
    launch_returned = threading.Event()
    unload_started = threading.Event()
    unload_done = threading.Event()
    dispatch_results = []
    unload_results = []
    handler_calls = []

    def build(**_kwargs):
        service_entered.set()
        assert release_service.wait(timeout=5)
        return SimpleNamespace(
            _subagent_id="sa-launch-unload",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        )

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    def handler(_args, *, invocation):
        handler_calls.append(True)
        handle = invocation.subagents.launch(SubagentLaunchRequest(goal="admitted"))
        launch_returned.set()
        return handle.subagent_id

    registration = context.register_tool(
        "_launch_unload_probe", "debugging",
        _schema("_launch_unload_probe"), handler,
    )
    retained_entry = registry.get_entry(
        "_launch_unload_probe", scope=manager.scope_key
    )
    assert retained_entry is not None

    def launch():
        with bind_subagent_parent(parent):
            dispatch_results.append(registry.dispatch(
                "_launch_unload_probe", {}, scope=manager.scope_key,
                session_id="launch-unload-session", task_id="turn-launch",
            ))

    def unload():
        unload_started.set()
        unload_results.append(manager.unload("launch-unload-owner"))
        unload_done.set()

    launch_thread = threading.Thread(target=launch, name="launch-before-unload")
    unload_thread = threading.Thread(target=unload, name="unload-after-launch")
    launch_thread.start()
    assert service_entered.wait(timeout=2)
    unload_thread.start()
    assert unload_started.wait(timeout=2)
    completed_while_launch_was_in_service = unload_done.wait(timeout=2.1)
    with bind_subagent_parent(parent):
        rejected = registry.dispatch(
            "_launch_unload_probe", {}, scope=manager.scope_key,
            session_id="launch-unload-session", task_id="turn-rejected",
        )
    with pytest.raises(SubagentLifecycleError, match="authority is unavailable"):
        with retained_entry._execution_scope_factory():
            raise AssertionError("closed authority entered a new execution scope")
    release_service.set()
    launch_thread.join(timeout=5)
    unload_thread.join(timeout=5)

    assert completed_while_launch_was_in_service is False
    assert launch_returned.is_set()
    assert dispatch_results == ["sa-launch-unload"]
    assert unload_results == [True]
    assert len(handler_calls) == 1
    assert "Unknown tool" in rejected
    assert registration is not None


def test_launch_after_completed_unload_fails_before_lifecycle_service(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "unload-first-owner")
    retained = []

    def handler(_args, *, invocation):
        retained.append(invocation.subagents)
        return "captured"

    registration = context.register_tool(
        "_unload_first_probe", "debugging",
        _schema("_unload_first_probe"), handler,
    )
    assert registry.dispatch(
        "_unload_first_probe", {}, scope=manager.scope_key,
        session_id="unload-first-session", task_id="turn-capture",
    ) == "captured"
    assert manager.unload("unload-first-owner") is True

    def forbidden_build(**_kwargs):
        raise AssertionError("revoked launch reached lifecycle service")

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", forbidden_build
    )
    with pytest.raises(Exception, match="authority is unavailable"):
        retained[0].launch(SubagentLaunchRequest(goal="must fail"))
    assert registration is not None


@pytest.mark.parametrize("control", ["steer", "stop", "collect"])
def test_admitted_control_finishes_before_unload_revokes_authority(
    tmp_path, monkeypatch, control
):
    context, manager = _context(tmp_path, "steer-unload-owner")
    parent = SimpleNamespace(
        session_id="steer-unload-session", enabled_toolsets=["file"]
    )
    child_release = threading.Event()
    authorization_complete = threading.Event()
    release_authorization = threading.Event()
    unload_done = threading.Event()
    events = []
    handles = []
    control_receipts = []

    child = SimpleNamespace(
        _subagent_id="sa-steer-unload",
        _delegate_role="leaf",
        _delegate_depth=1,
        provider="test",
        model="test-model",
    )

    def steer(_text):
        events.append("service")
        return True

    child.steer = steer
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: child,
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: (
            child_release.wait(timeout=5)
            and {
                "status": "completed",
                "summary": "done",
                "api_calls": 0,
                "duration_seconds": 0,
            }
        ),
    )
    if control == "stop":
        def stop(_service, _handle, *, reason):
            assert reason == "finish before unload"
            events.append("service")
            return lifecycle_module.SubagentCancelResult(True)

        monkeypatch.setattr(
            lifecycle_module.SubagentLifecycleService,
            "stop",
            stop,
        )
    elif control == "collect":
        def collect(_service, handle):
            events.append("service")
            result = lifecycle_module.SubagentResult(
                handle,
                SubagentState.SUCCEEDED,
                True,
                summary="done",
                completed_at=1.0,
                result_hash="stable-hash",
            )
            return lifecycle_module.SubagentCompletion(
                2,
                1,
                "stable-event",
                handle,
                True,
                SubagentState.SUCCEEDED,
                result,
                2.0,
            )

        monkeypatch.setattr(
            lifecycle_module.SubagentLifecycleService,
            "collect",
            collect,
        )

    def handler(args, *, invocation):
        if args["action"] == "launch":
            handles.append(
                invocation.subagents.launch(SubagentLaunchRequest(goal="stay active"))
            )
            return "launched"
        if control == "steer":
            receipt = invocation.subagents.steer(
                handles[0], "finish before unload"
            )
            result = receipt.disposition.value
        elif control == "stop":
            receipt = invocation.subagents.stop(
                handles[0], reason="finish before unload"
            )
            result = "accepted" if receipt.accepted else "rejected"
        else:
            receipt = invocation.subagents.collect(handles[0])
            result = "ready" if receipt.ready else "pending"
        control_receipts.append(receipt)
        return result

    registration = context.register_tool(
        "_steer_unload_probe",
        "debugging",
        _schema("_steer_unload_probe"),
        handler,
    )
    with bind_subagent_parent(parent):
        assert registry.dispatch(
            "_steer_unload_probe",
            {"action": "launch"},
            scope=manager.scope_key,
            session_id="steer-unload-session",
            task_id="turn-launch",
        ) == "launched"

    original_authorize = plugin_invocation_module._OwnerStore.authorize

    def authorize_with_barrier(owner_store, handle, session_id):
        authorized = original_authorize(owner_store, handle, session_id)
        if authorized:
            authorization_complete.set()
            assert release_authorization.wait(timeout=5)
        return authorized

    monkeypatch.setattr(
        plugin_invocation_module._OwnerStore,
        "authorize",
        authorize_with_barrier,
    )

    def dispatch_steer():
        with bind_subagent_parent(parent):
            return registry.dispatch(
                "_steer_unload_probe",
                {"action": "steer"},
                scope=manager.scope_key,
                session_id="steer-unload-session",
                task_id="turn-steer",
            )

    def unload():
        manager.unload("steer-unload-owner")
        events.append("unload")
        unload_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        steer_future = executor.submit(dispatch_steer)
        assert authorization_complete.wait(timeout=2)
        unload_future = executor.submit(unload)
        unload_completed_before_admitted_control = unload_done.wait(timeout=0.5)
        release_authorization.set()
        steer_result = steer_future.result(timeout=5)
        unload_future.result(timeout=5)

    child_release.set()
    assert unload_completed_before_admitted_control is False
    if control == "steer":
        assert steer_result == lifecycle_module.SubagentControlDisposition.QUEUED.value
    elif control == "stop":
        assert steer_result == "accepted"
    else:
        assert steer_result == "ready"
    if control != "collect":
        assert control_receipts[0].accepted is True
    assert events == ["service", "unload"]
    assert registration is not None


def test_concurrent_launch_and_authorize_share_one_atomic_owner_store(
    tmp_path, monkeypatch
):
    context, manager = _context(tmp_path, "concurrent-owner")
    parent = SimpleNamespace(session_id="concurrent-session", enabled_toolsets=["file"])
    counter = iter(range(20))
    counter_lock = threading.Lock()

    def build(**_kwargs):
        with counter_lock:
            ident = next(counter)
        return SimpleNamespace(
            _subagent_id=f"sa-concurrent-{ident}",
            _delegate_role="leaf",
            _delegate_depth=1,
            provider="test",
            model="test-model",
        )

    monkeypatch.setattr("tools.delegate_tool._build_child_preserving_parent_tools", build)
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args, **_kwargs: {
            "status": "completed", "summary": "done", "api_calls": 0,
            "duration_seconds": 0,
        },
    )

    def handler(args, *, invocation):
        handle = invocation.subagents.launch(
            SubagentLaunchRequest(goal=f"task-{args['index']}")
        )
        state = invocation.subagents.wait(handle, timeout_seconds=1).state
        return f"{handle.subagent_id}:{state.value}"

    registration = context.register_tool(
        "_concurrent_owner_probe", "debugging",
        _schema("_concurrent_owner_probe"), handler,
    )

    def launch(index):
        with bind_subagent_parent(parent):
            return registry.dispatch(
                "_concurrent_owner_probe", {"index": index},
                scope=manager.scope_key, session_id="concurrent-session",
                task_id=f"turn-{index}",
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(launch, range(16)))
    assert len({result.split(":", 1)[0] for result in results}) == 16
    assert all(result.endswith(":" + SubagentState.SUCCEEDED.value) for result in results)
    assert registration is not None
    registration.dispose()


def test_public_agent_resolution_preserves_complete_tools_prompt_and_toolsets(
    tmp_path, monkeypatch
):
    from agent.system_prompt import build_system_prompt
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
    context, manager = _context(tmp_path)
    parent = SimpleNamespace(session_id="session-async", enabled_toolsets=["file"])
    schema = _schema("_async_invocation_probe")
    toolsets_before = list(parent.enabled_toolsets)

    def real_agent():
        return AIAgent(
            api_key="synthetic-test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            provider="openrouter",
            platform="cli",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="prompt-byte-session",
            enabled_toolsets=["plugin_invocation_probe"],
        )

    async def legacy_handler(args, **kwargs):
        await asyncio.sleep(0)
        return "legacy"

    legacy_registration = context.register_tool(
        "_async_invocation_probe",
        "plugin_invocation_probe",
        schema,
        legacy_handler,
        is_async=True,
    )
    legacy_agent = real_agent()
    legacy_tools = json.dumps(
        legacy_agent.tools, ensure_ascii=False, sort_keys=True
    ).encode()
    legacy_model_prompt = legacy_agent._format_tools_for_system_message().encode()
    legacy_system_prompt = build_system_prompt(legacy_agent).encode()
    legacy_enabled_toolsets = list(legacy_agent.enabled_toolsets)
    assert legacy_registration is not None
    legacy_registration.dispose()
    opted_agent = None

    async def handler(args, *, invocation, **kwargs):
        await asyncio.sleep(0)
        return json.dumps({
            "session": invocation.session_id,
            "task": invocation.task_id,
            "kwargs": sorted(kwargs),
        })

    registration = context.register_tool(
        "_async_invocation_probe",
        "plugin_invocation_probe",
        schema,
        handler,
        is_async=True,
    )
    try:
        opted_agent = real_agent()
        opted_tools_before = json.dumps(
            opted_agent.tools, ensure_ascii=False, sort_keys=True
        ).encode()
        opted_model_prompt_before = (
            opted_agent._format_tools_for_system_message().encode()
        )
        opted_system_prompt_before = build_system_prompt(opted_agent).encode()
        assert opted_tools_before == legacy_tools
        assert opted_model_prompt_before == legacy_model_prompt
        assert opted_system_prompt_before == legacy_system_prompt
        assert opted_agent.enabled_toolsets == legacy_enabled_toolsets
        with bind_subagent_parent(parent):
            result = json.loads(registry.dispatch(
                "_async_invocation_probe",
                {},
                scope=manager.scope_key,
                session_id="session-async",
                task_id="turn-async",
            ))
        assert result == {
            "session": "session-async",
            "task": "turn-async",
            "kwargs": ["session_id", "task_id"],
        }
        assert json.dumps(
            opted_agent.tools, ensure_ascii=False, sort_keys=True
        ).encode() == opted_tools_before
        assert (
            opted_agent._format_tools_for_system_message().encode()
            == opted_model_prompt_before
        )
        assert build_system_prompt(opted_agent).encode() == opted_system_prompt_before
        assert parent.enabled_toolsets == toolsets_before
    finally:
        if opted_agent is not None:
            opted_agent.close()
        legacy_agent.close()
        assert registration is not None
        registration.dispose()


def test_opt_in_preserves_registry_exception_contract(tmp_path):
    context, manager = _context(tmp_path)

    def legacy(args, **kwargs):
        raise RuntimeError(f"same-{args['value']}-{kwargs['task_id']}")

    def opted_in(args, *, invocation, **kwargs):
        assert invocation.task_id == kwargs["task_id"]
        raise RuntimeError(f"same-{args['value']}-{kwargs['task_id']}")

    legacy_registration = context.register_tool(
        "_legacy_exception_probe", "debugging",
        _schema("_legacy_exception_probe"), legacy,
    )
    opted_registration = context.register_tool(
        "_opted_exception_probe", "debugging",
        _schema("_opted_exception_probe"), opted_in,
    )
    try:
        legacy_result = registry.dispatch(
            "_legacy_exception_probe", {"value": 7}, scope=manager.scope_key,
            session_id="session", task_id="turn",
        )
        opted_result = registry.dispatch(
            "_opted_exception_probe", {"value": 7}, scope=manager.scope_key,
            session_id="session", task_id="turn",
        )
        assert legacy_result == opted_result
        error = json.loads(legacy_result)["error"]
        assert "RuntimeError" in error
        assert "same-7-turn" in error
    finally:
        assert legacy_registration is not None
        assert opted_registration is not None
        legacy_registration.dispose()
        opted_registration.dispose()
