"""Long-lived CLI fallback-chain and configured-primary regressions."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import yaml

from hermes_cli.auth import AuthError
from hermes_cli.model_switch import ModelSwitchResult


_OPUS = "claude-opus-5"
_ANTHROPIC = "anthropic"
_OLD_CHAIN = [
    {"provider": "openai-codex", "model": "gpt-5.6-sol"},
    {"provider": "xai-oauth", "model": "grok-4.6"},
]
_DESIRED_CHAIN = [
    {"provider": "xai-oauth", "model": "grok-4.6"},
    {"provider": "openai-codex", "model": "gpt-5.6-sol"},
]
_SELECTED_REASONING = {"enabled": True, "effort": "medium"}


class _SwitchAgent:
    _config_context_length = None
    _custom_providers = None

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.before_failure = None
        self.reasoning_config = {"enabled": True, "effort": "low"}
        self.calls = []

    def switch_model(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            if self.before_failure is not None:
                self.before_failure()
            raise RuntimeError("selected transport failed")
        self.reasoning_config = copy.deepcopy(_SELECTED_REASONING)


def _startup_fallback_switch_shell(*, fail: bool = False):
    return SimpleNamespace(
        model="gpt-5.6-sol",
        provider="openai-codex",
        requested_provider="openai-codex",
        reasoning_config={"enabled": True, "effort": "low"},
        _configured_primary_runtime={
            "model": _OPUS,
            "provider": _ANTHROPIC,
            "reasoning_config": {"enabled": True, "effort": "high"},
        },
        _startup_auth_fallback_active=True,
        _explicit_api_key="codex-test-key",
        _explicit_base_url="https://chatgpt.com/backend-api/codex",
        api_key="codex-test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        agent=_SwitchAgent(fail=fail),
        conversation_history=[],
        _pending_model_switch_note=None,
        _pending_one_turn_model_restore=None,
        _session_db=None,
        session_id=None,
        _confirm_expensive_model_switch=lambda _result: True,
    )


def _selected_result():
    return ModelSwitchResult(
        success=True,
        new_model="claude-sonnet-4.6",
        target_provider="anthropic",
        api_key="anthropic-selected-key",
        base_url="https://api.anthropic.com",
        api_mode="anthropic_messages",
        provider_label="Anthropic",
    )


def _patch_switch_display(monkeypatch):
    import cli as cli_module

    monkeypatch.setattr(cli_module, "_cprint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_module, "save_config_value", lambda *_args: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli_module.HermesCLI,
        "_clear_persisted_context_for_model_switch",
        lambda *_args, **_kwargs: None,
    )
    return cli_module


def _make_shell(monkeypatch, tmp_path, *, chain):
    import cli as cli_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setitem(
        cli_module.CLI_CONFIG,
        "model",
        {"default": _OPUS, "provider": _ANTHROPIC},
    )
    monkeypatch.setitem(
        cli_module.CLI_CONFIG,
        "agent",
        {"reasoning_effort": "high"},
    )
    monkeypatch.setitem(cli_module.CLI_CONFIG, "fallback_providers", copy.deepcopy(chain))
    monkeypatch.setitem(cli_module.CLI_CONFIG, "fallback_model", None)
    shell = cli_module.HermesCLI(compact=True, max_turns=1)
    assert shell.model == _OPUS
    assert shell.requested_provider == _ANTHROPIC
    assert shell.reasoning_config == {"enabled": True, "effort": "high"}
    return cli_module, shell


def _write_chain(home, chain) -> None:
    (home / "config.yaml").write_text(
        yaml.safe_dump({"fallback_providers": chain}, sort_keys=False),
        encoding="utf-8",
    )


def test_long_lived_cli_refreshes_fallback_order_before_next_turn(
    tmp_path, monkeypatch
):
    _, shell = _make_shell(monkeypatch, tmp_path, chain=_OLD_CHAIN)
    active_agent = SimpleNamespace(
        _fallback_chain=copy.deepcopy(_OLD_CHAIN),
        _fallback_model=copy.deepcopy(_OLD_CHAIN[0]),
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys={"stale"},
    )
    shell.agent = active_agent

    _write_chain(tmp_path, _DESIRED_CHAIN)
    refreshed = shell._refresh_fallback_model()

    assert refreshed == _DESIRED_CHAIN
    assert shell._fallback_model == _DESIRED_CHAIN
    assert active_agent._fallback_chain == _DESIRED_CHAIN
    assert active_agent._fallback_model == _DESIRED_CHAIN[0]
    assert active_agent._fallback_index == 0
    assert active_agent._unavailable_fallback_keys == set()


def test_startup_auth_fallback_preserves_primary_and_retries_it_next_turn(
    tmp_path, monkeypatch
):
    cli_module, shell = _make_shell(monkeypatch, tmp_path, chain=_OLD_CHAIN)
    _write_chain(tmp_path, _DESIRED_CHAIN)
    requested = []

    def resolve_runtime_provider(**kwargs):
        provider = kwargs.get("requested")
        requested.append(provider)
        if provider == _ANTHROPIC and requested.count(_ANTHROPIC) == 1:
            raise AuthError("primary unavailable", provider=_ANTHROPIC)
        if provider == "xai-oauth":
            raise AuthError("grok unavailable", provider="xai-oauth")
        if provider == "openai-codex":
            return {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "codex-test-key",
                "credential_pool": None,
                "source": "test",
            }
        if provider == _ANTHROPIC:
            return {
                "provider": _ANTHROPIC,
                "api_mode": "anthropic_messages",
                "base_url": "https://api.anthropic.com",
                "api_key": "anthropic-test-key",
                "credential_pool": None,
                "source": "test",
            }
        raise AssertionError(f"unexpected provider {provider!r}")

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key",
        lambda _entry: None,
    )
    monkeypatch.setattr(shell, "_normalize_model_for_provider", lambda _provider: False)

    class FakeAgent:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self._primary_runtime = {
                "model": kwargs["model"],
                "provider": kwargs["provider"],
                "requested_provider": kwargs["requested_provider"],
                "reasoning_config": copy.deepcopy(kwargs["reasoning_config"]),
            }

    monkeypatch.setattr(cli_module, "AIAgent", FakeAgent)
    monkeypatch.setattr(shell, "_install_tool_callbacks", lambda: None)
    monkeypatch.setattr(shell, "_ensure_tirith_security", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )

    assert shell._init_agent() is True
    assert requested == [_ANTHROPIC, "xai-oauth", "openai-codex"]
    assert shell.model == "gpt-5.6-sol"
    assert shell.provider == "openai-codex"
    assert shell._configured_primary_runtime == {
        "model": _OPUS,
        "provider": _ANTHROPIC,
        "reasoning_config": {"enabled": True, "effort": "high"},
    }

    fallback_agent = shell.agent
    assert fallback_agent._primary_runtime == {
        "model": _OPUS,
        "provider": _ANTHROPIC,
        "requested_provider": _ANTHROPIC,
        "reasoning_config": {"enabled": True, "effort": "high"},
        "restorable": False,
    }

    assert shell._ensure_runtime_credentials() is True
    assert requested == [
        _ANTHROPIC,
        "xai-oauth",
        "openai-codex",
        _ANTHROPIC,
    ]
    assert shell.model == _OPUS
    assert shell.provider == _ANTHROPIC
    assert shell.requested_provider == _ANTHROPIC
    assert shell.reasoning_config == {"enabled": True, "effort": "high"}


def test_torn_cli_config_keeps_last_known_good_chain(tmp_path, monkeypatch):
    _, shell = _make_shell(monkeypatch, tmp_path, chain=_DESIRED_CHAIN)
    (tmp_path / "config.yaml").write_text(
        "fallback_providers: [\n", encoding="utf-8"
    )

    assert shell._refresh_fallback_model() == _DESIRED_CHAIN
    assert shell._fallback_model == _DESIRED_CHAIN


def test_nonrestorable_startup_primary_is_not_mixed_with_fallback_transport():
    from agent.agent_runtime_helpers import restore_primary_runtime

    agent = SimpleNamespace(
        _fallback_activated=True,
        _fallback_index=2,
        _primary_runtime={
            "model": _OPUS,
            "provider": _ANTHROPIC,
            "restorable": False,
        },
        model="gpt-5.6-sol",
        provider="openai-codex",
    )

    assert restore_primary_runtime(agent) is False
    assert agent.model == "gpt-5.6-sol"
    assert agent.provider == "openai-codex"
    assert agent._fallback_index == 2


def test_picker_switch_replaces_startup_primary_and_clears_latch(monkeypatch):
    cli_module = _patch_switch_display(monkeypatch)
    shell = _startup_fallback_switch_shell()

    cli_module.HermesCLI._apply_model_switch_result(
        shell, _selected_result(), persist_global=False,
    )

    assert shell.reasoning_config == _SELECTED_REASONING
    assert shell._configured_primary_runtime == {
        "model": "claude-sonnet-4.6",
        "provider": "anthropic",
        "reasoning_config": _SELECTED_REASONING,
    }
    assert shell._startup_auth_fallback_active is False


@pytest.mark.parametrize("persist_global", [False, True], ids=["session", "global"])
def test_typed_switch_replaces_startup_primary_for_session_and_global(
    monkeypatch, persist_global
):
    cli_module = _patch_switch_display(monkeypatch)
    shell = _startup_fallback_switch_shell()

    cli_module.HermesCLI._confirm_and_apply_cli_model_switch(
        shell,
        _selected_result(),
        persist_global=persist_global,
        one_turn=False,
    )

    assert shell.reasoning_config == _SELECTED_REASONING
    assert shell._configured_primary_runtime == {
        "model": "claude-sonnet-4.6",
        "provider": "anthropic",
        "reasoning_config": _SELECTED_REASONING,
    }
    assert shell._startup_auth_fallback_active is False


def test_typed_once_preserves_configured_primary_and_startup_latch(monkeypatch):
    cli_module = _patch_switch_display(monkeypatch)
    shell = _startup_fallback_switch_shell()
    configured_before = copy.deepcopy(shell._configured_primary_runtime)
    shell._snapshot_model_runtime = (
        cli_module.HermesCLI._snapshot_model_runtime.__get__(shell)
    )

    cli_module.HermesCLI._confirm_and_apply_cli_model_switch(
        shell, _selected_result(), persist_global=False, one_turn=True,
    )

    assert shell._configured_primary_runtime == configured_before
    assert shell._startup_auth_fallback_active is True
    assert shell._pending_one_turn_model_restore["reasoning_config"] == {
        "enabled": True,
        "effort": "low",
    }
    assert (
        shell._pending_one_turn_model_restore["_configured_primary_runtime"]
        == configured_before
    )
    assert (
        shell._pending_one_turn_model_restore["_startup_auth_fallback_active"]
        is True
    )


@pytest.mark.parametrize("switch_path", ["picker", "typed"])
def test_failed_switch_restores_primary_snapshot_and_latch(
    monkeypatch, switch_path
):
    cli_module = _patch_switch_display(monkeypatch)
    shell = _startup_fallback_switch_shell(fail=True)
    configured_before = copy.deepcopy(shell._configured_primary_runtime)
    reasoning_before = copy.deepcopy(shell.reasoning_config)

    def _simulate_partial_switch_state():
        shell.reasoning_config = copy.deepcopy(_SELECTED_REASONING)
        shell._configured_primary_runtime = {
            "model": "partially-committed-model",
            "provider": "partially-committed-provider",
            "reasoning_config": copy.deepcopy(_SELECTED_REASONING),
        }
        shell._startup_auth_fallback_active = False

    shell.agent.before_failure = _simulate_partial_switch_state

    if switch_path == "picker":
        cli_module.HermesCLI._apply_model_switch_result(
            shell, _selected_result(), persist_global=False,
        )
    else:
        cli_module.HermesCLI._confirm_and_apply_cli_model_switch(
            shell, _selected_result(), persist_global=False, one_turn=False,
        )

    assert shell.reasoning_config == reasoning_before
    assert shell._configured_primary_runtime == configured_before
    assert shell._startup_auth_fallback_active is True
