"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``key_env`` names the env var
    holding the key; ``api_key_env`` accepted as an alias). Returns None when
    neither yields a non-empty value, letting ``resolve_runtime_provider``
    fall through to the provider's standard credential resolution.

    ``key_env`` is resolved through ``agent.secret_scope.get_secret`` rather
    than a raw ``os.getenv`` — in a multiplexed gateway a bare env read would
    ignore the active profile's scope and can return another profile's
    credential. ``get_secret`` already implements the right fallback: it
    reads ``os.environ`` when there's no active multiplexed scope (matching
    prior single-profile behavior), and fails closed only when multiplexing
    is active with no scope installed.
    """
    if not isinstance(entry, dict):
        return None
    inline = str(entry.get("api_key") or "").strip()
    if inline:
        return inline
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        from agent.secret_scope import get_secret

        return (get_secret(key_env) or "").strip() or None
    return None


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain


def refresh_fallback_chain(
    config_path: Path,
    previous_chain: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Read one complete fallback-chain snapshot from ``config_path``.

    A missing config is an authoritative empty chain. A transient read or
    parse failure keeps a fresh copy of the last known-good chain. Managed
    overlays and environment expansion match the gateway runtime loader.
    """

    previous = [dict(entry) for entry in (previous_chain or [])]
    try:
        if not config_path.exists():
            return []
        from hermes_cli.config import read_user_config_raw

        config = read_user_config_raw(config_path)
        try:
            from hermes_cli import managed_scope

            config = managed_scope.apply_managed_overlay(config)
        except Exception:
            pass
        try:
            from hermes_cli.config import _expand_env_vars

            expanded = _expand_env_vars(config)
            if isinstance(expanded, dict):
                config = expanded
        except Exception:
            pass
    except Exception:
        return previous
    return get_fallback_chain(config)


def apply_fallback_chain_to_agent(
    agent: Any,
    chain: list[dict[str, Any]] | None,
) -> None:
    """Apply a whole chain between turns without disrupting live cooldown."""

    if agent is None:
        return
    new_chain = [dict(entry) for entry in (chain or [])]
    rate_limited_until = getattr(agent, "_rate_limited_until", 0) or 0
    if (
        getattr(agent, "_fallback_activated", False)
        and rate_limited_until > time.monotonic()
    ):
        return
    old_chain = list(getattr(agent, "_fallback_chain", []) or [])
    agent._fallback_chain = new_chain
    agent._fallback_model = new_chain[0] if new_chain else None
    if not getattr(agent, "_fallback_activated", False):
        agent._fallback_index = 0
    if new_chain != old_chain:
        unavailable = getattr(agent, "_unavailable_fallback_keys", None)
        if unavailable:
            unavailable.clear()
