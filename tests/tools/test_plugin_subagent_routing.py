"""Behavioral contract for host-owned plugin subagent route adaptation."""

from __future__ import annotations

import os
import dataclasses
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import agent.subagent_lifecycle as lifecycle_contracts
from agent.secret_scope import reset_secret_scope, set_secret_scope
from agent.secret_scope import (
    reset_authoritative_secret_scope,
    set_authoritative_secret_scope,
)
from agent.subagent_lifecycle import (
    SubagentLaunchRequest,
    SubagentLaunchRequestV2,
    SubagentLifecycleError,
    SubagentLifecycleService,
    bind_subagent_parent,
)
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools import delegate_tool
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


def test_route_catalog_and_assessment_are_bounded_immutable_and_launch_free(
    monkeypatch,
):
    parent = _parent()
    service = SubagentLifecycleService(lambda: parent)
    monkeypatch.setattr(
        "tools.delegate_tool._catalog_subagent_routes",
        lambda _parent: (
            ("synthetic", "model-a"),
            ("synthetic", "model-b"),
        ),
    )
    _install_route(monkeypatch)
    monkeypatch.setattr(
        "tools.delegate_tool._resolved_exact_empty_model_tools", lambda: ()
    )
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **_kwargs: pytest.fail("catalog/assessment must not build a child"),
    )

    catalog = service.catalog_routes()
    assert isinstance(catalog, lifecycle_contracts.SubagentRouteCatalog)
    assert catalog.complete is True
    assert catalog.candidate_count == 2
    assert isinstance(catalog.assessed_at, float)
    assert catalog.assessed_at > 0
    assert re.fullmatch(r"snap_[0-9a-f]{32}", catalog.snapshot_id)
    assert catalog.routes == (
        lifecycle_contracts.SubagentRouteIdentity("synthetic", "model-a"),
        lifecycle_contracts.SubagentRouteIdentity("synthetic", "model-b"),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        catalog.complete = False

    assessment = service.assess_route("synthetic", "model-a")
    assert isinstance(assessment, lifecycle_contracts.SubagentRouteAssessment)
    assert assessment.route == lifecycle_contracts.SubagentRouteIdentity(
        "synthetic", "model-a"
    )
    assert assessment.eligible is True
    assert assessment.reason == "ELIGIBLE"
    assert assessment.transport == "chat_completions"
    assert assessment.authenticated is True
    assert assessment.agent_capable is True
    assert assessment.exact_empty_model_tools is True
    assert assessment.mutation_evidence_complete is True
    assert assessment.independent_mutation_channels == frozenset()
    assert assessment.hermes_model_tool_count == 0
    assert isinstance(assessment.assessed_at, float)
    assert assessment.assessed_at > 0
    assert re.fullmatch(r"asm_[0-9a-f]{32}", assessment.assessment_id)
    assert "secret" not in repr(assessment).lower()


def test_route_catalog_and_assessment_fail_closed_without_reflecting_canaries(
    monkeypatch,
):
    secret = "sk-route-catalog-secret-canary"
    url = "https://route-secret.invalid/private"
    path = "/private/route/credentials.json"
    parent = _parent()
    service = SubagentLifecycleService(lambda: parent)
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: (_ for _ in ()).throw(RuntimeError(f"{secret} {url} {path}")),
    )

    catalog = service.catalog_routes()
    assert catalog.complete is False
    assert catalog.routes == ()
    assert catalog.candidate_count == 0
    assert catalog.reason == "CATALOG_UNAVAILABLE"
    assert catalog.assessed_at > 0
    assert re.fullmatch(r"snap_[0-9a-f]{32}", catalog.snapshot_id)

    monkeypatch.setattr(
        "tools.delegate_tool._resolve_delegation_credentials",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError(f"{secret} {url} {path}")
        ),
    )
    assessment = service.assess_route("synthetic", "model-a")
    assert assessment.eligible is False
    assert assessment.reason == "ROUTE_UNAVAILABLE"
    assert assessment.transport == "unavailable"
    assert assessment.authenticated is False
    assert assessment.agent_capable is False
    assert assessment.exact_empty_model_tools is False
    assert assessment.mutation_evidence_complete is False
    assert assessment.assessed_at > 0
    assert re.fullmatch(r"asm_[0-9a-f]{32}", assessment.assessment_id)
    public = repr((catalog, assessment))
    assert secret not in public
    assert url not in public
    assert path not in public

    with pytest.raises(SubagentLifecycleError) as url_error:
        service.assess_route(url, "model-a")
    assert url not in str(url_error.value)


def test_route_catalog_marks_unverified_custom_or_partial_inventory_incomplete(
    monkeypatch,
):
    parent = _parent()
    service = SubagentLifecycleService(lambda: parent)
    monkeypatch.setattr(
        "tools.delegate_tool._catalog_subagent_routes",
        lambda _parent: (_ for _ in ()).throw(
            OverflowError("configured route inventory is incomplete")
        ),
    )

    catalog = service.catalog_routes()

    assert catalog.complete is False
    assert catalog.routes == ()
    assert catalog.reason == "CATALOG_INCOMPLETE"


def test_public_route_identifiers_enforce_exact_storage_boundary():
    boundary = "m" * 200
    assert lifecycle_contracts.SubagentRouteIdentity("synthetic", boundary).model == boundary

    with pytest.raises(SubagentLifecycleError):
        lifecycle_contracts.SubagentRouteIdentity("synthetic", "m" * 201)


@pytest.mark.parametrize("field", ["provider", "model"])
def test_route_assessment_preserves_exact_200_character_identity(
    monkeypatch, field
):
    boundary = "r" * 200
    route = {
        "provider": boundary if field == "provider" else "synthetic",
        "model": boundary if field == "model" else "model-a",
        "api_mode": "chat_completions",
    }
    monkeypatch.setattr(
        "tools.delegate_tool._resolved_exact_empty_model_tools", lambda: ()
    )

    receipt = delegate_tool._assess_native_read_only_route(route)

    assert getattr(receipt, field) == boundary
    assert receipt.eligible is True


@pytest.mark.parametrize("field", ["provider", "model"])
def test_route_assessment_rejects_201_character_identity(monkeypatch, field):
    route = {
        "provider": "r" * 201 if field == "provider" else "synthetic",
        "model": "r" * 201 if field == "model" else "model-a",
        "api_mode": "chat_completions",
    }
    monkeypatch.setattr(
        "tools.delegate_tool._resolved_exact_empty_model_tools", lambda: ()
    )

    with pytest.raises(ValueError, match="bounded public identifier"):
        delegate_tool._assess_native_read_only_route(route)


def test_offline_catalog_requires_authoritative_scope_and_ignores_process_env(
    monkeypatch, tmp_path,
):
    descriptors = [
        SimpleNamespace(
            slug="openrouter",
            auth_type="api_key",
            api_key_env_vars=("OPENROUTER_API_KEY",),
            keyless=False,
        ),
        SimpleNamespace(
            slug="openai-api",
            auth_type="api_key",
            api_key_env_vars=("OPENAI_API_KEY",),
            keyless=False,
        ),
    ]
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog", lambda: descriptors
    )
    monkeypatch.setattr(
        "hermes_cli.models._PROVIDER_MODELS",
        {"openai-api": ["gpt-safe"]},
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
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **_kwargs: pytest.fail("offline catalog must not call legacy picker"),
    )
    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda *_args, **_kwargs: pytest.fail("offline catalog must not seed pools"),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "process-poison-canary")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with pytest.raises(ValueError, match="authoritative"):
        delegate_tool._catalog_subagent_routes(_parent())

    token = set_authoritative_secret_scope({"OPENAI_API_KEY": "profile-key"})
    try:
        with pytest.raises(OverflowError, match="incomplete"):
            delegate_tool._catalog_subagent_routes(
                _parent(provider="openrouter", model="poison-model")
            )
        assert delegate_tool._catalog_subagent_routes(
            _parent(provider="openai-api", model="gpt-safe")
        ) == (
            ("openai-api", "gpt-safe"),
        )
    finally:
        reset_authoritative_secret_scope(token)


@pytest.mark.parametrize(
    "raw_process_credentials",
    [False, True],
)
def test_bedrock_assessment_requires_authoritative_profile_credentials(
    monkeypatch, tmp_path, raw_process_credentials
):
    model = "us.anthropic.claude-sonnet-4-6"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if raw_process_credentials:
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "raw-process-poison")
    else:
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_PROFILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("agent.bedrock_adapter.has_aws_credentials", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog",
        lambda: [SimpleNamespace(
            slug="bedrock", auth_type="aws_sdk",
            api_key_env_vars=(), keyless=False,
        )],
    )
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {"bedrock": [model]})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "bedrock",
            "model": model,
            "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "api_key": "raw-process-poison" if raw_process_credentials else "aws-sdk",
            "api_mode": "bedrock_converse",
            "request_overrides": {},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "recognized": True,
            "corrected_model": None,
        },
    )
    monkeypatch.setattr(
        "tools.delegate_tool._resolved_exact_empty_model_tools", lambda: ()
    )
    token = set_authoritative_secret_scope({})
    try:
        with pytest.raises(OverflowError, match="incomplete"):
            delegate_tool._catalog_subagent_routes(
                _parent(provider="bedrock", model=model)
            )
        assessment = SubagentLifecycleService(lambda: _parent()).assess_route(
            "bedrock", model
        )
    finally:
        reset_authoritative_secret_scope(token)

    assert assessment.reason == "ROUTE_UNAVAILABLE"
    assert assessment.authenticated is False
    assert assessment.agent_capable is False
    assert assessment.eligible is False


def test_bedrock_bound_profile_credentials_require_private_worker_bundle(
    monkeypatch, tmp_path
):
    model = "us.anthropic.claude-sonnet-4-6"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            with_overrides=lambda **_kwargs: SimpleNamespace(
                user_providers={}, custom_providers=[], excluded_providers=[]
            )
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog",
        lambda: [SimpleNamespace(
            slug="bedrock", auth_type="aws_sdk",
            api_key_env_vars=(), keyless=False,
        )],
    )
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {"bedrock": [model]})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "bedrock", "model": model,
            "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "api_key": "aws-sdk", "api_mode": "bedrock_converse",
            "request_overrides": {},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": True, "recognized": True, "corrected_model": None,
        },
    )
    monkeypatch.setattr(
        "tools.delegate_tool._resolved_exact_empty_model_tools", lambda: ()
    )
    parent = _parent(provider="bedrock", model=model)
    token = set_authoritative_secret_scope({
        "AWS_ACCESS_KEY_ID": "profile-access-key",
        "AWS_SECRET_ACCESS_KEY": "profile-secret-key",
        "AWS_SESSION_TOKEN": "profile-session-token",
    })
    try:
        with pytest.raises(OverflowError, match="incomplete"):
            delegate_tool._catalog_subagent_routes(parent)
        assessment = SubagentLifecycleService(lambda: parent).assess_route(
            "bedrock", model
        )
    finally:
        reset_authoritative_secret_scope(token)

    assert assessment.reason == "ROUTE_UNAVAILABLE"
    assert assessment.eligible is False
    assert assessment.authenticated is False
    assert assessment.agent_capable is False


def test_concurrent_poisoned_bedrock_profiles_never_assess_or_launch(
    monkeypatch, tmp_path
):
    model = "us.anthropic.claude-sonnet-4-6"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "raw-default-bearer-poison")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "raw-default-access-poison")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "raw-default-secret-poison")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "raw-default-session-poison")
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog",
        lambda: [SimpleNamespace(
            slug="bedrock", auth_type="aws_sdk",
            api_key_env_vars=(), keyless=False,
        )],
    )
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {"bedrock": [model]})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "bedrock", "model": model,
            "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "api_key": "aws-sdk", "api_mode": "bedrock_converse",
            "request_overrides": {},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": True, "recognized": True, "corrected_model": None,
        },
    )
    monkeypatch.setattr(
        "tools.delegate_tool._resolved_exact_empty_model_tools", lambda: ()
    )
    build_calls = []
    monkeypatch.setattr(
        "tools.delegate_tool._build_child_preserving_parent_tools",
        lambda **kwargs: build_calls.append(kwargs),
    )
    scopes = (
        {"AWS_BEARER_TOKEN_BEDROCK": "profile-a-bearer-canary"},
        {
            "AWS_ACCESS_KEY_ID": "profile-b-access-canary",
            "AWS_SECRET_ACCESS_KEY": "profile-b-secret-canary",
            "AWS_SESSION_TOKEN": "profile-b-session-canary",
        },
    )

    def probe(index, scope):
        parent = _parent(
            provider="bedrock", model=model, session_id=f"bedrock-{index}"
        )
        service = SubagentLifecycleService(lambda: parent)
        request = SubagentLaunchRequestV2(
            api_contract_version=2,
            base=SubagentLaunchRequest(goal="read only", model=model),
            toolset_mode="exact",
            exact_toolsets=(),
            provider="bedrock",
        )
        token = set_authoritative_secret_scope(scope)
        try:
            assessment = service.assess_route("bedrock", model)
            with pytest.raises(SubagentLifecycleError) as caught:
                service.launch(request)
            return assessment, str(caught.value)
        finally:
            reset_authoritative_secret_scope(token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = tuple(pool.map(lambda item: probe(*item), enumerate(scopes)))

    assert all(receipt.reason == "ROUTE_UNAVAILABLE" for receipt, _ in receipts)
    assert all(receipt.eligible is False for receipt, _ in receipts)
    assert all(
        error == "Requested provider/model route is unavailable."
        for _, error in receipts
    )
    assert build_calls == []
    assert delegation_admission.active_background_units() == 0


def test_vertex_catalog_and_assessment_share_unprovable_offline_auth(
    monkeypatch, tmp_path
):
    model = "google/gemini-3.1-pro-preview"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            with_overrides=lambda **_kwargs: SimpleNamespace(
                user_providers={}, custom_providers=[], excluded_providers=[]
            )
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog",
        lambda: [SimpleNamespace(
            slug="vertex", auth_type="vertex", api_key_env_vars=(), keyless=False,
        )],
    )
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {"vertex": [model]})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "vertex", "model": model,
            "base_url": "https://vertex.invalid/v1",
            "api_key": "minted-but-unproven-token",
            "api_mode": "chat_completions", "request_overrides": {},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *_args, **_kwargs: {
            "accepted": True, "recognized": True, "corrected_model": None,
        },
    )
    monkeypatch.setattr(
        "tools.delegate_tool._resolved_exact_empty_model_tools", lambda: ()
    )
    parent = _parent(provider="vertex", model=model)
    token = set_authoritative_secret_scope({
        "VERTEX_CREDENTIALS_PATH": "/profile/credential-canary.json",
        "VERTEX_PROJECT_ID": "profile-project",
    })
    try:
        with pytest.raises(OverflowError, match="incomplete"):
            delegate_tool._catalog_subagent_routes(parent)
        assessment = SubagentLifecycleService(lambda: parent).assess_route(
            "vertex", model
        )
    finally:
        reset_authoritative_secret_scope(token)

    assert assessment.reason == "ROUTE_UNAVAILABLE"
    assert assessment.authenticated is False
    assert assessment.agent_capable is False


def test_offline_catalog_does_not_repair_malformed_profile_auth(
    monkeypatch, tmp_path
):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{malformed-secret-canary", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            with_overrides=lambda **_kwargs: SimpleNamespace(
                user_providers={}, custom_providers=[], excluded_providers=[]
            )
        ),
    )
    token = set_authoritative_secret_scope({})
    try:
        with pytest.raises(OverflowError, match="incomplete"):
            delegate_tool._catalog_subagent_routes(_parent())
    finally:
        reset_authoritative_secret_scope(token)

    assert auth_path.read_text(encoding="utf-8") == "{malformed-secret-canary"
    assert not (tmp_path / "auth.json.corrupt").exists()


@pytest.mark.parametrize("fresh", [True, False])
def test_offline_catalog_requires_fresh_profile_local_oauth(
    monkeypatch, tmp_path, fresh
):
    expires_at = time.time() + (3600 if fresh else -3600)
    (tmp_path / "auth.json").write_text(
        json.dumps({
            "providers": {
                "openai-codex": {
                    "source": "device_code",
                    "access_token": "profile-oauth-canary",
                    "expires_at": expires_at,
                }
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: SimpleNamespace(
            with_overrides=lambda **_kwargs: SimpleNamespace(
                user_providers={}, custom_providers=[], excluded_providers=[]
            )
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog",
        lambda: [SimpleNamespace(
            slug="openai-codex",
            auth_type="oauth_external",
            api_key_env_vars=(),
            keyless=False,
        )],
    )
    monkeypatch.setattr(
        "hermes_cli.models._PROVIDER_MODELS", {"openai-codex": ["gpt-safe"]}
    )
    token = set_authoritative_secret_scope({})
    try:
        if fresh:
            assert delegate_tool._catalog_subagent_routes(
                _parent(provider="openai-codex", model="gpt-safe")
            ) == (
                ("openai-codex", "gpt-safe"),
            )
        else:
            with pytest.raises(OverflowError, match="incomplete"):
                delegate_tool._catalog_subagent_routes(
                    _parent(provider="openai-codex", model="gpt-safe")
                )
    finally:
        reset_authoritative_secret_scope(token)


def _profile_tree_snapshot(root):
    snapshot = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            snapshot.append((relative, "symlink", os.readlink(path)))
        elif path.is_file():
            snapshot.append((relative, "file", path.read_bytes(), path.stat().st_mode))
        else:
            snapshot.append((relative, "dir", path.stat().st_mode))
    return snapshot


@pytest.mark.parametrize("malformed", [False, True])
def test_offline_catalog_never_mutates_fresh_profile_tree(
    monkeypatch, tmp_path, malformed
):
    profile = tmp_path / "profile"
    profile.mkdir()
    if malformed:
        (profile / "config.yaml").write_text("{malformed", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(
        "hermes_cli.inventory.load_picker_context",
        lambda: pytest.fail("offline catalog must not use mutating config loader"),
    )
    monkeypatch.setattr("hermes_cli.provider_catalog.provider_catalog", lambda: [])
    before = _profile_tree_snapshot(profile)
    token = set_authoritative_secret_scope({})
    try:
        if malformed:
            with pytest.raises(OverflowError, match="incomplete"):
                delegate_tool._catalog_subagent_routes(_parent(provider="", model=""))
        else:
            assert delegate_tool._catalog_subagent_routes(
                _parent(provider="", model="")
            ) == ()
    finally:
        reset_authoritative_secret_scope(token)

    assert _profile_tree_snapshot(profile) == before


def test_offline_catalog_rejects_config_final_symlink(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    other = tmp_path / "other-config.yaml"
    other.write_text("model: {provider: openai-api}\n", encoding="utf-8")
    (profile / "config.yaml").symlink_to(other)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    before = _profile_tree_snapshot(tmp_path)
    token = set_authoritative_secret_scope({})
    try:
        with pytest.raises(OverflowError, match="incomplete"):
            delegate_tool._catalog_subagent_routes(_parent(provider="", model=""))
    finally:
        reset_authoritative_secret_scope(token)
    assert _profile_tree_snapshot(tmp_path) == before


def test_offline_catalog_rejects_config_final_component_swap(monkeypatch, tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    config_path = profile / "config.yaml"
    original_path = profile / "config.original.yaml"
    config_path.write_text("model: {provider: openai-api}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    real_open = delegate_tool.os.open
    swapped = False

    def swap_then_open(path, flags, *args):
        nonlocal swapped
        if str(path) == str(config_path) and not swapped:
            swapped = True
            os.replace(config_path, original_path)
            config_path.write_text("providers: {victim: {model: stolen}}\n", encoding="utf-8")
        return real_open(path, flags, *args)

    monkeypatch.setattr(delegate_tool.os, "open", swap_then_open)
    token = set_authoritative_secret_scope({})
    try:
        with pytest.raises(OverflowError, match="incomplete"):
            delegate_tool._catalog_subagent_routes(_parent(provider="", model=""))
    finally:
        reset_authoritative_secret_scope(token)
    assert swapped is True


def test_offline_catalog_rejects_non_string_excluded_provider(monkeypatch, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model_catalog:\n  excluded_providers:\n    - 7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog",
        lambda: [SimpleNamespace(
            slug="opencode-free",
            auth_type="api_key",
            api_key_env_vars=(),
            keyless=True,
        )],
    )
    monkeypatch.setattr(
        "hermes_cli.models._PROVIDER_MODELS", {"opencode-free": ["model-a"]}
    )
    service = SubagentLifecycleService(
        lambda: _parent(provider="opencode-free", model="model-a")
    )
    token = set_authoritative_secret_scope({})
    try:
        catalog = service.catalog_routes()
    finally:
        reset_authoritative_secret_scope(token)

    assert catalog.complete is False
    assert catalog.routes == ()
    assert catalog.candidate_count == 0
    assert catalog.reason == "CATALOG_INCOMPLETE"


@pytest.mark.parametrize(
    ("provider", "auth_type", "env_names", "keyless", "scope"),
    [
        ("opencode-free", "api_key", (), True, {}),
        ("openai-api", "api_key", ("OPENAI_API_KEY",), False,
         {"OPENAI_API_KEY": "scoped-key-canary"}),
        ("bedrock", "aws_sdk", (), False, {
            "AWS_ACCESS_KEY_ID": "scoped-access-canary",
            "AWS_SECRET_ACCESS_KEY": "scoped-secret-canary",
        }),
    ],
)
def test_malformed_profile_auth_dominates_positive_provider_evidence(
    monkeypatch, tmp_path, provider, auth_type, env_names, keyless, scope
):
    model = "model-a"
    (tmp_path / "auth.json").write_text(
        json.dumps({"providers": {provider: "malformed-entry"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog",
        lambda: [SimpleNamespace(
            slug=provider,
            auth_type=auth_type,
            api_key_env_vars=env_names,
            keyless=keyless,
        )],
    )
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {provider: [model]})
    service = SubagentLifecycleService(
        lambda: _parent(provider=provider, model=model)
    )
    token = set_authoritative_secret_scope(scope)
    try:
        catalog = service.catalog_routes()
    finally:
        reset_authoritative_secret_scope(token)

    assert catalog.complete is False
    assert catalog.routes == ()
    assert catalog.candidate_count == 0
    assert catalog.reason == "CATALOG_INCOMPLETE"


@pytest.mark.parametrize(
    ("provider", "auth_type", "env_names", "keyless", "scope"),
    [
        ("opencode-free", "api_key", (), True, {}),
        ("openai-api", "api_key", ("OPENAI_API_KEY",), False,
         {"OPENAI_API_KEY": "scoped-key-canary"}),
        ("bedrock", "aws_sdk", (), False, {
            "AWS_ACCESS_KEY_ID": "scoped-access-canary",
            "AWS_SECRET_ACCESS_KEY": "scoped-secret-canary",
        }),
    ],
)
def test_malformed_profile_config_dominates_positive_provider_evidence(
    monkeypatch, tmp_path, provider, auth_type, env_names, keyless, scope
):
    model = "model-a"
    (tmp_path / "config.yaml").write_text(
        f"{provider}: malformed-entry\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.provider_catalog.provider_catalog",
        lambda: [SimpleNamespace(
            slug=provider,
            auth_type=auth_type,
            api_key_env_vars=env_names,
            keyless=keyless,
        )],
    )
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {provider: [model]})
    service = SubagentLifecycleService(
        lambda: _parent(provider=provider, model=model)
    )
    token = set_authoritative_secret_scope(scope)
    try:
        catalog = service.catalog_routes()
    finally:
        reset_authoritative_secret_scope(token)

    assert catalog.complete is False
    assert catalog.routes == ()
    assert catalog.candidate_count == 0
    assert catalog.reason == "CATALOG_INCOMPLETE"


@pytest.mark.parametrize("clock_value", [float("nan"), float("inf"), -1.0])
def test_public_route_receipt_timestamp_stays_bounded_on_invalid_host_clock(
    monkeypatch, clock_value
):
    monkeypatch.setattr(lifecycle_contracts.time, "time", lambda: clock_value)

    catalog = lifecycle_contracts._new_route_catalog(
        complete=True, routes=(), candidate_count=0, reason="COMPLETE"
    )

    assert 0 < catalog.assessed_at <= 253_402_300_799.0


@pytest.mark.parametrize(
    ("model", "complete"),
    [("m" * 200, True), ("m" * 201, False)],
)
def test_route_catalog_enforces_exact_storage_identifier_boundary(
    monkeypatch, model, complete
):
    parent = _parent()
    service = SubagentLifecycleService(lambda: parent)
    def catalog(_parent):
        if len(model) > 200:
            raise OverflowError("route inventory is incomplete")
        return (("synthetic", model),)

    monkeypatch.setattr("tools.delegate_tool._catalog_subagent_routes", catalog)

    catalog = service.catalog_routes()

    assert catalog.complete is complete
    if complete:
        assert catalog.routes == (
            lifecycle_contracts.SubagentRouteIdentity("synthetic", model),
        )
    else:
        assert catalog.routes == ()
        assert catalog.reason == "CATALOG_INCOMPLETE"


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

    with pytest.raises(ValueError, match="bounded public identifier"):
        _assess_native_read_only_route(
            {
                "provider": "p" * 201,
                "model": "m" * 201,
                "api_mode": "chat_completions",
                "command": None,
                "args": [],
            }
        )

    receipt = _assess_native_read_only_route(
        {
            "provider": "p" * 200,
            "model": "m" * 200,
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
        }
    )
    assert len(receipt.provider) == 200
    assert len(receipt.model) == 200
    assert receipt.transport == "chat_completions"
    assert receipt.mutation_evidence_complete is True
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


def test_route_assessment_exposes_incomplete_exact_empty_evidence_fail_closed(
    monkeypatch,
):
    _install_route(monkeypatch)
    monkeypatch.setattr(
        "tools.delegate_tool._resolved_exact_empty_model_tools",
        lambda: (_ for _ in ()).throw(RuntimeError("private resolver detail")),
    )

    assessment = SubagentLifecycleService(lambda: _parent()).assess_route(
        "synthetic", "model-a"
    )

    assert assessment.eligible is False
    assert assessment.reason == "MUTATION_CHANNEL_UNAVAILABLE"
    assert assessment.transport == "chat_completions"
    assert assessment.authenticated is True
    assert assessment.agent_capable is True
    assert assessment.exact_empty_model_tools is False
    assert assessment.mutation_evidence_complete is False
    assert assessment.independent_mutation_channels == frozenset()
    assert assessment.hermes_model_tool_count == 0


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
