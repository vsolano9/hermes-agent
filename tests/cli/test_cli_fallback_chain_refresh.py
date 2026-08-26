"""Long-lived CLI fallback-chain and configured-primary regressions."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import yaml

from hermes_cli.auth import AuthError


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
