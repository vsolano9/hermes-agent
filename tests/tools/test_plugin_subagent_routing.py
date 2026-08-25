"""Behavioral contract for host-owned plugin subagent route adaptation."""

from __future__ import annotations

import os
import dataclasses
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLaunchRequestV2,
    SubagentLifecycleError,
    SubagentLifecycleService,
    bind_subagent_parent,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools import delegation_admission
from tools.registry import registry


class _Child:
    def __init__(self, ident: str, kwargs: dict):
        self._subagent_id = ident
        self._delegate_role = kwargs.get("role", "leaf")
        self._delegate_depth = 1
        self.provider = kwargs.get("override_provider") or "parent-provider"
        self.model = kwargs.get("model") or "parent-model"
        self.api_mode = kwargs.get("override_api_mode") or "chat_completions"
        self.acp_command = kwargs.get("override_acp_command")
        self.acp_args = list(kwargs.get("override_acp_args") or [])
        self.valid_tool_names = set()
        self.tools = []


def _parent(**overrides):
    values = {
        "session_id": "routing-parent",
        "enabled_toolsets": ["file", "terminal", "delegation"],
        "disabled_toolsets": [],
        "provider": "parent-provider",
        "model": "parent-model",
        "api_mode": "chat_completions",
        "base_url": "https://parent.invalid/v1",
        "api_key": "parent-secret",
        "acp_command": None,
        "acp_args": [],
        "reasoning_config": {"enabled": True, "effort": "low"},
        "_delegate_depth": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _reset_admission():
    delegation_admission._reset_for_tests()
    yield
    delegation_admission._reset_for_tests()


def _install_child_seams(monkeypatch, captured):
    counter = iter(range(100))

    def build(**kwargs):
        captured.append(dict(kwargs))
        return _Child(f"sa-route-{next(counter)}", kwargs)

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: {
            "status": "completed",
            "summary": "owner result",
            "api_calls": 1,
            "duration_seconds": 0.01,
        },
    )


def _install_route(monkeypatch, *, provider="synthetic", mode="chat_completions"):
    route = {
        "provider": provider,
        "model": "model-a",
        "base_url": "https://profile-a.invalid/v1",
        "api_key": "profile-a-secret",
        "api_mode": mode,
        "request_overrides": {"safe": True},
        "max_output_tokens": 2048,
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: dict(route),
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "persist": False,
            "recognized": True,
            "message": None,
        },
    )
    return route


def _routed_request(*, mode="exact", toolsets=(), reasoning="high", workdir=None):
    return SubagentLaunchRequestV2(
        api_contract_version=2,
        base=SubagentLaunchRequest(
            goal="read only inspection",
            model="model-a",
            working_directory=workdir,
        ),
        toolset_mode=mode,
        exact_toolsets=toolsets,
        provider="synthetic",
        reasoning_effort=reasoning,
    )


def test_v2_route_and_reasoning_use_host_resolution_without_public_secrets(
    monkeypatch,
):
    captured = []
    _install_child_seams(monkeypatch, captured)
    route = _install_route(monkeypatch)
    parent = _parent()
    before_cwd = os.getcwd()
    before_toolsets = list(parent.enabled_toolsets)
    service = SubagentLifecycleService(lambda: parent)

    handle = service.launch(_routed_request())
    assert service.wait(handle, timeout_seconds=1).completed is True

    assert captured[0]["model"] == "model-a"
    assert captured[0]["override_provider"] == "synthetic"
    assert captured[0]["override_base_url"] == route["base_url"]
    assert captured[0]["override_api_key"] == route["api_key"]
    assert captured[0]["override_reasoning_config"] == {
        "enabled": True,
        "effort": "high",
    }
    assert captured[0]["toolsets"] == []
    assert captured[0]["exact_toolsets"] is True
    public = handle.to_dict()
    assert public["provider"] == "synthetic"
    assert public["model"] == "model-a"
    assert not ({"api_key", "base_url", "api_mode", "request_overrides"} & public.keys())
    assert route["api_key"] not in repr(handle)
    assert route["base_url"] not in repr(handle)
    assert parent.enabled_toolsets == before_toolsets
    assert os.getcwd() == before_cwd


@pytest.mark.parametrize(
    ("failure", "launch_request"),
    [
        ("route", _routed_request()),
        ("reasoning", _routed_request(reasoning="impossible-level")),
        ("workdir", _routed_request(workdir="/tmp/forbidden-canary")),
    ],
)
def test_invalid_route_reasoning_and_workdir_fail_before_admission_and_build(
    monkeypatch, failure, launch_request
):
    import model_tools

    captured = []
    _install_child_seams(monkeypatch, captured)
    key_canary = "sk-profile-canary-not-public"
    url_canary = "https://secret-profile.invalid/private"
    path_canary = "/private/profile/config.yaml"
    parent = _parent()
    parent_toolsets = list(parent.enabled_toolsets)
    parent_cwd = os.getcwd()
    monkeypatch.setattr(
        model_tools, "_last_resolved_tool_names", ["parent-visible-canary"]
    )

    if failure == "route":
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError(f"{key_canary} {url_canary} {path_canary}")
            ),
        )
    else:
        _install_route(monkeypatch)

    with pytest.raises(SubagentLifecycleError) as caught:
        SubagentLifecycleService(lambda: parent).launch(launch_request)

    diagnostic = str(caught.value)
    assert all(canary not in diagnostic for canary in (key_canary, url_canary, path_canary))
    assert len(diagnostic) <= 160
    assert captured == []
    assert delegation_admission.active_background_units() == 0
    assert parent.enabled_toolsets == parent_toolsets
    assert os.getcwd() == parent_cwd
    assert model_tools._last_resolved_tool_names == ["parent-visible-canary"]


@pytest.mark.parametrize("failure_stage", ["route", "build"])
def test_route_and_build_failures_drop_raw_exception_chains_and_log_canaries(
    monkeypatch, caplog, tmp_path, failure_stage
):
    canaries = (
        "sk-raw-exception-canary",
        "https://raw-exception.invalid/private",
        "RAW_ENV_CANARY=value",
        "/private/raw/config.yaml",
        "dangerous-worker --write",
    )
    raw_message = " ".join(canaries)
    parent = _parent(session_id=f"raw-{failure_stage}")

    if failure_stage == "route":
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(raw_message)),
        )
    else:
        _install_route(monkeypatch)
        monkeypatch.setattr(
            "tools.delegate_tool._build_child_preserving_parent_tools",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(raw_message)),
        )

    with pytest.raises(SubagentLifecycleError) as caught:
        SubagentLifecycleService(lambda: parent).launch(_routed_request())

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert all(canary not in str(caught.value) for canary in canaries)
    assert delegation_admission.active_background_units() == 0

    home = tmp_path / failure_stage
    home.mkdir()
    manager = PluginManager(scope_key=str(home.resolve()))
    context = PluginContext(
        PluginManifest(
            name=f"raw-{failure_stage}",
            key=f"raw-{failure_stage}",
            source="user",
        ),
        manager,
    )
    tool_name = f"_raw_route_{failure_stage}"

    def handler(_args, *, invocation):
        invocation.subagents.launch(_routed_request())

    registration = context.register_tool(
        tool_name,
        "debugging",
        {
            "name": tool_name,
            "description": "raw exception probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler,
    )
    caplog.set_level(logging.ERROR, logger="tools.registry")
    try:
        with bind_subagent_parent(parent):
            result = registry.dispatch(
                tool_name,
                {},
                scope=manager.scope_key,
                session_id=parent.session_id,
                task_id="raw-error-turn",
            )
    finally:
        assert registration is not None
        registration.dispose()

    assert "SubagentLifecycleError" in result
    assert all(canary not in result for canary in canaries)
    assert all(canary not in caplog.text for canary in canaries)
    assert delegation_admission.active_background_units() == 0


@pytest.mark.parametrize(
    "validation",
    [
        {"accepted": False, "recognized": False},
        {"accepted": True, "recognized": False},
        {
            "accepted": True,
            "recognized": True,
            "corrected_model": "model-a-corrected",
        },
    ],
)
def test_unavailable_or_unverified_model_fails_before_admission_and_build(
    monkeypatch, validation
):
    captured = []
    _install_child_seams(monkeypatch, captured)
    _install_route(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: dict(
            validation,
            persist=False,
            message="hostile https://url.invalid /path KEY_CANARY",
        ),
    )

    with pytest.raises(SubagentLifecycleError) as caught:
        SubagentLifecycleService(lambda: _parent()).launch(_routed_request())

    assert str(caught.value) == "Requested provider/model route is unavailable."
    assert captured == []
    assert delegation_admission.active_background_units() == 0


def test_provider_only_route_validates_parent_model_before_build(monkeypatch):
    captured = []
    validated = []
    _install_child_seams(monkeypatch, captured)
    route = _install_route(monkeypatch)
    route["model"] = "parent-model"
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: (
            pytest.fail("wrong target model")
            if kwargs["target_model"] != "parent-model"
            else dict(route)
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda model, provider, **_kwargs: (
            validated.append((model, provider))
            or {"accepted": True, "recognized": True}
        ),
    )
    request = SubagentLaunchRequestV2(
        api_contract_version=2,
        base=SubagentLaunchRequest(goal="provider default model"),
        toolset_mode="inherit",
        provider="synthetic",
    )

    handle = SubagentLifecycleService(lambda: _parent()).launch(request)

    assert handle.model == "parent-model"
    assert validated == [("parent-model", "synthetic")]
    assert captured[0]["model"] == "parent-model"


def test_real_opencode_free_exact_route_accepts_native_anonymous_runtime(
    monkeypatch,
):
    from hermes_cli import models as model_catalog
    from hermes_cli import runtime_provider

    captured = []
    _install_child_seams(monkeypatch, captured)
    native_resolver = runtime_provider.resolve_runtime_provider
    validation_calls = []
    native_validator = model_catalog.validate_requested_model

    def resolve_anonymous(**kwargs):
        route = native_resolver(**kwargs)
        assert route["provider"] == "opencode-free"
        route["api_key"] = ""  # native anonymous contract: no bearer
        return route

    def validate_exact(model, provider, **kwargs):
        validation_calls.append((model, provider, kwargs.get("api_key")))
        return native_validator(model, provider, **kwargs)

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", resolve_anonymous
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model", validate_exact
    )
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda *_a, **_k: None)
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "environment-key-poison")
    parent = _parent(
        provider="openrouter",
        model="parent-model",
        api_key="parent-key-poison",
    )
    request = SubagentLaunchRequestV2(
        api_contract_version=2,
        base=SubagentLaunchRequest(
            goal="anonymous read-only inspection",
            model="x-preview-f-free",
        ),
        toolset_mode="exact",
        exact_toolsets=(),
        provider="opencode-free",
    )

    service = SubagentLifecycleService(lambda: parent)
    handle = service.launch(request)
    assert service.wait(handle, timeout_seconds=1).completed is True

    assert validation_calls == [
        ("x-preview-f-free", "opencode-free", "")
    ]
    assert captured[0]["override_provider"] == "opencode-free"
    assert captured[0]["override_api_key"] == ""
    assert captured[0]["authoritative_route_overrides"] is True
    assert captured[0]["exact_toolsets"] is True
    assert captured[0]["toolsets"] == []
    assert handle.provider == "opencode-free"
    assert handle.model == "x-preview-f-free"
    assert "api_key" not in handle.to_dict()
    assert "parent-key-poison" not in repr(handle)
    assert "environment-key-poison" not in repr(handle)


def test_route_resolution_is_profile_scoped_and_collection_does_not_reresolve(
    monkeypatch, tmp_path,
):
    captured = []
    resolutions = []
    release = threading.Event()
    worker_scopes = []
    _install_child_seams(monkeypatch, captured)

    from agent.secret_scope import get_secret

    def resolve_runtime_provider(**kwargs):
        resolutions.append((kwargs, get_secret("ROUTE_KEY"), get_secret("ROUTE_URL")))
        return {
            "provider": kwargs["requested"],
            "model": kwargs["target_model"],
            "base_url": get_secret("ROUTE_URL"),
            "api_key": get_secret("ROUTE_KEY"),
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {"accepted": True, "recognized": True},
    )
    def run_child(*_args):
        from hermes_constants import get_hermes_home

        release.wait(1)
        worker_scopes.append((get_secret("ROUTE_KEY"), str(get_hermes_home())))
        return {
            "status": "completed",
            "summary": "profile-bound owner content",
            "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run_child)
    service = SubagentLifecycleService(lambda: _parent())

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()

    token_a = set_secret_scope(
        {"ROUTE_KEY": "profile-a-key", "ROUTE_URL": "https://a.invalid/v1"}
    )
    home_token_a = set_hermes_home_override(str(home_a))
    try:
        handle_a = service.launch(_routed_request())
    finally:
        reset_hermes_home_override(home_token_a)
        reset_secret_scope(token_a)

    token_b = set_secret_scope(
        {"ROUTE_KEY": "profile-b-key", "ROUTE_URL": "https://b.invalid/v1"}
    )
    home_token_b = set_hermes_home_override(str(home_b))
    try:
        handle_b = service.launch(_routed_request().__class__(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="profile b", model="model-a"),
            toolset_mode="exact",
            exact_toolsets=(),
            provider="synthetic",
            reasoning_effort="high",
        ))
        release.set()
        assert service.collect(handle_a).handle == handle_a
        assert service.wait(handle_a, timeout_seconds=1).completed is True
        assert service.wait(handle_b, timeout_seconds=1).completed is True
    finally:
        reset_hermes_home_override(home_token_b)
        reset_secret_scope(token_b)

    assert [item[1:] for item in resolutions] == [
        ("profile-a-key", "https://a.invalid/v1"),
        ("profile-b-key", "https://b.invalid/v1"),
    ]
    assert captured[0]["override_api_key"] == "profile-a-key"
    assert captured[0]["override_base_url"] == "https://a.invalid/v1"
    assert captured[1]["override_api_key"] == "profile-b-key"
    assert captured[1]["override_base_url"] == "https://b.invalid/v1"
    assert len(resolutions) == 2
    assert set(worker_scopes) == {
        ("profile-a-key", str(home_a)),
        ("profile-b-key", str(home_b)),
    }


@pytest.mark.parametrize(
    ("provider", "api_mode", "command", "eligible"),
    [
        ("synthetic", "chat_completions", None, True),
        ("synthetic", "codex_responses", None, True),
        ("synthetic", "anthropic_messages", None, True),
        ("synthetic", "bedrock_converse", None, True),
        ("synthetic", "unknown_transport", None, False),
        ("synthetic", "chat_completions", "external-worker", False),
        ("copilot-acp", "chat_completions", None, False),
    ],
)
def test_exact_empty_native_read_only_transport_is_allowlisted_and_fail_closed(
    monkeypatch, provider, api_mode, command, eligible, tmp_path
):
    captured = []
    _install_child_seams(monkeypatch, captured)
    canary = tmp_path / "must-not-be-written"
    runtime = _install_route(monkeypatch, provider=provider, mode=api_mode)
    runtime["command"] = command
    runtime["args"] = ["writeTextFile", str(canary)] if command else []
    if command:
        monkeypatch.setattr("shutil.which", lambda _name: "/synthetic/bin/worker")
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: dict(runtime),
    )

    service = SubagentLifecycleService(lambda: _parent())
    if eligible:
        handle = service.launch(_routed_request())
        assert service.wait(handle, timeout_seconds=1).completed is True
        assert len(captured) == 1
    else:
        with pytest.raises(SubagentLifecycleError) as caught:
            service.launch(_routed_request())
        assert str(caught.value) == "Native read-only transport is unavailable."
        assert captured == []
        assert delegation_admission.active_background_units() == 0
    assert not canary.exists()


def test_transport_gate_does_not_reinterpret_routed_inherit_or_exact_nonempty(
    monkeypatch,
):
    captured = []
    _install_child_seams(monkeypatch, captured)
    _install_route(monkeypatch, provider="copilot-acp")
    service = SubagentLifecycleService(lambda: _parent())

    for mode, toolsets in (("inherit", ()), ("exact", ("file",))):
        handle = service.launch(_routed_request(mode=mode, toolsets=toolsets))
        assert service.wait(handle, timeout_seconds=1).completed is True

    assert [call["exact_toolsets"] for call in captured] == [False, True]


def test_reasoning_omission_preserves_native_configured_or_parent_path(monkeypatch):
    captured = []
    _install_child_seams(monkeypatch, captured)
    _install_route(monkeypatch)
    request = _routed_request(reasoning=None)

    handle = SubagentLifecycleService(lambda: _parent()).launch(request)
    assert "override_reasoning_config" not in captured[0]
    assert handle.provider == "synthetic"


@pytest.mark.parametrize("invalid", [True, "", "   ", "turbo"])
def test_reasoning_invalid_or_ambiguous_values_fail_closed(monkeypatch, invalid):
    captured = []
    _install_child_seams(monkeypatch, captured)
    _install_route(monkeypatch)

    with pytest.raises(SubagentLifecycleError):
        SubagentLifecycleService(lambda: _parent()).launch(
            _routed_request(reasoning=invalid)
        )

    assert captured == []
    assert delegation_admission.active_background_units() == 0


def test_reasoning_explicit_false_disables_without_inheriting(monkeypatch):
    captured = []
    _install_child_seams(monkeypatch, captured)
    _install_route(monkeypatch)

    handle = SubagentLifecycleService(lambda: _parent()).launch(
        _routed_request(reasoning=False)
    )

    assert handle.provider == "synthetic"
    assert captured[0]["override_reasoning_config"] == {"enabled": False}


def test_exact_empty_inherited_acp_transport_fails_before_build(monkeypatch):
    captured = []
    _install_child_seams(monkeypatch, captured)
    parent = _parent(
        provider="copilot-acp",
        api_mode="chat_completions",
        acp_command="copilot",
        acp_args=["write_text_file"],
    )
    request = SubagentLaunchRequestV2(
        api_contract_version=2,
        base=SubagentLaunchRequest(goal="try mutation canary"),
        toolset_mode="exact",
        exact_toolsets=(),
    )

    with pytest.raises(SubagentLifecycleError) as caught:
        SubagentLifecycleService(lambda: parent).launch(request)

    assert str(caught.value) == "Native read-only transport is unavailable."
    assert captured == []
    assert delegation_admission.active_background_units() == 0


def test_instantiated_mutation_channel_mismatch_is_disposed_before_worker(
    monkeypatch,
):
    _install_route(monkeypatch)
    built = []
    ran = []

    class UnsafeChild(_Child):
        def __init__(self, kwargs):
            super().__init__("sa-unsafe-effective", kwargs)
            self.valid_tool_names = {"write_file"}
            self.tools = [
                {
                    "type": "function",
                    "function": {"name": "write_file", "parameters": {}},
                }
            ]
            self.closed = False

        def close(self):
            self.closed = True

    def build(**kwargs):
        child = UnsafeChild(kwargs)
        built.append(child)
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: ran.append(True),
    )

    with pytest.raises(SubagentLifecycleError) as caught:
        SubagentLifecycleService(lambda: _parent()).launch(_routed_request())

    assert str(caught.value) == "Native read-only transport is unavailable."
    assert len(built) == 1
    assert built[0].closed is True
    assert ran == []
    assert delegation_admission.active_background_units() == 0


def test_native_read_only_receipt_binds_bounded_model_and_model_mismatch_cleans_up(
    monkeypatch,
):
    from tools.delegate_tool import _assess_native_read_only_route

    receipt = _assess_native_read_only_route(
        {
            "provider": "p" * 1024,
            "model": "m" * 1024,
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
        }
    )
    assert len(receipt.provider) <= 128
    assert len(receipt.model) <= 128
    assert receipt.transport == "chat_completions"
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.model = "forged-model"

    _install_route(monkeypatch)
    built = []
    ran = []

    class WrongModelChild(_Child):
        def __init__(self, kwargs):
            super().__init__("sa-wrong-effective-model", kwargs)
            self.model = "different-model"
            self.closed = False

        def close(self):
            self.closed = True

    def build(**kwargs):
        child = WrongModelChild(kwargs)
        built.append(child)
        return child

    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr(
        "tools.delegate_tool._run_child_lifecycle",
        lambda *_args: ran.append(True),
    )

    with pytest.raises(SubagentLifecycleError) as caught:
        SubagentLifecycleService(lambda: _parent()).launch(_routed_request())

    assert str(caught.value) == "Native read-only transport is unavailable."
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert len(built) == 1
    assert built[0].closed is True
    assert ran == []
    assert delegation_admission.active_background_units() == 0


def test_bound_plugin_launch_anchors_two_concurrent_manager_profiles(
    monkeypatch, tmp_path
):
    from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    homes = {name: tmp_path / name for name in ("profile-a", "profile-b")}
    for name, home in homes.items():
        home.mkdir()
        (home / ".env").write_text(
            f"ROUTE_KEY={name}-key\nROUTE_URL=https://{name}.invalid/v1\n",
            encoding="utf-8",
        )
    process_poison = "process-only-poison-key"
    monkeypatch.setenv("MISSING_ROUTE_SECRET", process_poison)

    barrier = threading.Barrier(2)
    resolved = []
    built = []
    workers = []

    def runtime(**kwargs):
        barrier.wait(timeout=2)
        snapshot = (
            str(get_hermes_home()),
            get_secret("ROUTE_KEY"),
            get_secret("ROUTE_URL"),
            get_secret("MISSING_ROUTE_SECRET"),
        )
        resolved.append(snapshot)
        return {
            "provider": kwargs["requested"],
            "model": kwargs["target_model"],
            "base_url": snapshot[2],
            "api_key": snapshot[1],
            "api_mode": "chat_completions",
        }

    child_counter = iter(range(10))

    def build(**kwargs):
        built.append(
            (
                str(get_hermes_home()),
                get_secret("ROUTE_KEY"),
                kwargs["override_base_url"],
                kwargs["override_api_key"],
                get_secret("MISSING_ROUTE_SECRET"),
            )
        )
        return _Child(f"sa-bound-profile-{next(child_counter)}", kwargs)

    def run(*_args):
        workers.append(
            (
                str(get_hermes_home()),
                get_secret("ROUTE_KEY"),
                get_secret("MISSING_ROUTE_SECRET"),
            )
        )
        return {
            "status": "completed",
            "summary": "owner result",
            "api_calls": 0,
            "duration_seconds": 0,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", runtime
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {"accepted": True, "recognized": True},
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools", build
    )
    monkeypatch.setattr("tools.delegate_tool._run_child_lifecycle", run)

    registrations = []
    dispatches = []
    for index, (name, home) in enumerate(homes.items()):
        manager = PluginManager(scope_key=str(home.resolve()))
        context = PluginContext(
            PluginManifest(
                name=f"route-{name}", key=f"route-{name}", source="user"
            ),
            manager,
        )
        tool_name = f"_route_profile_{index}"

        def handler(_args, *, invocation):
            handle = invocation.subagents.launch(_routed_request())
            invocation.subagents.wait(handle, timeout_seconds=2)
            return handle.provider

        registrations.append(
            context.register_tool(
                tool_name,
                "debugging",
                {
                    "name": tool_name,
                    "description": "profile route probe",
                    "parameters": {"type": "object", "properties": {}},
                },
                handler,
            )
        )
        dispatches.append((tool_name, manager.scope_key, name))

    def dispatch(item):
        tool_name, scope, name = item
        poison_home = tmp_path / f"poison-{name}"
        poison_home.mkdir()
        home_token = set_hermes_home_override(str(poison_home))
        secret_token = set_secret_scope(
            {"ROUTE_KEY": "poison-key", "ROUTE_URL": "https://poison.invalid"}
        )
        try:
            parent = _parent(session_id=f"session-{name}")
            with bind_subagent_parent(parent):
                return registry.dispatch(
                    tool_name,
                    {},
                    scope=scope,
                    session_id=parent.session_id,
                    task_id=f"turn-{name}",
                )
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(dispatch, dispatches)) == ["synthetic", "synthetic"]
    finally:
        for registration in registrations:
            assert registration is not None
            registration.dispose()

    expected_resolved = {
        (str(home), f"{name}-key", f"https://{name}.invalid/v1", None)
        for name, home in homes.items()
    }
    assert set(resolved) == expected_resolved
    assert set(built) == {
        (
            str(home),
            f"{name}-key",
            f"https://{name}.invalid/v1",
            f"{name}-key",
            None,
        )
        for name, home in homes.items()
    }
    assert set(workers) == {
        (str(home), f"{name}-key", None) for name, home in homes.items()
    }
    assert all(
        process_poison not in repr(receipt) and "poison" not in repr(receipt)
        for receipt in (resolved, built, workers)
    )
