from __future__ import annotations

import logging
from types import MappingProxyType

import pytest

from agent.delegation_context import delegated_child_context
from agent.system_prompt import _plugin_session_info
from hermes_cli.plugins import (
    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS,
    PluginContext,
    PluginManager,
    PluginManifest,
)


def _context(manager: PluginManager, name: str = "example-plugin") -> PluginContext:
    return PluginContext(
        PluginManifest(name=name, key=name, source="user"),
        manager,
    )


def test_registration_validates_stable_id_position_budget_and_duplicates():
    manager = PluginManager()
    ctx = _context(manager)

    for invalid_id in ("", "UPPER.case", "has space", "line\nbreak", "x" * 129):
        with pytest.raises(ValueError):
            ctx.register_system_prompt_section(invalid_id, "content")

    with pytest.raises(ValueError):
        ctx.register_system_prompt_section("example.rules", "content", position="priority-17")
    with pytest.raises(ValueError):
        ctx.register_system_prompt_section("example.rules", "content", max_chars=0)

    ctx.register_system_prompt_section("example.rules", "content")
    with pytest.raises(ValueError, match="already registered"):
        _context(manager, "other-plugin").register_system_prompt_section(
            "example.rules", "other content"
        )


def test_render_is_deterministic_bounded_and_session_info_is_read_only(caplog):
    manager = PluginManager()
    ctx = _context(manager)
    observed = []

    def render_b(info):
        observed.append(info)
        with pytest.raises(TypeError):
            info["session_id"] = "changed"
        return "B"

    ctx.register_system_prompt_section("example.z", render_b, max_chars=4)
    ctx.register_system_prompt_section("example.a", "A", max_chars=4)
    ctx.register_system_prompt_section("example.too-large", "12345", max_chars=4)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        rendered = manager.render_system_prompt_sections({"session_id": "session-1"})

    assert [(item.id, item.content) for item in rendered] == [
        ("example.a", "A"),
        ("example.z", "B"),
    ]
    assert isinstance(observed[0], MappingProxyType)
    assert observed[0]["session_id"] == "session-1"
    assert "exceeded max_chars" in caplog.text


def test_render_fails_open_for_callback_failure_wrong_type_and_aggregate_budget(caplog):
    manager = PluginManager()
    ctx = _context(manager)

    def boom(_info):
        raise RuntimeError("plugin exploded")

    ctx.register_system_prompt_section("example.boom", boom)
    ctx.register_system_prompt_section("example.wrong", lambda _info: {"not": "text"})
    chunk = (MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS // 2) - 200
    ctx.register_system_prompt_section("example.first", "a" * chunk, max_chars=chunk)
    ctx.register_system_prompt_section("example.second", "b" * chunk, max_chars=chunk)
    ctx.register_system_prompt_section("example.zzlast", "c" * 500, max_chars=500)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        rendered = manager.render_system_prompt_sections({})

    assert [item.id for item in rendered] == ["example.first", "example.second"]
    assert "plugin exploded" in caplog.text
    assert "returned dict, not str" in caplog.text
    assert "aggregate" in caplog.text


def test_root_scoped_prompt_section_is_absent_from_delegated_session():
    manager = PluginManager()
    ctx = _context(manager)
    ctx.register_system_prompt_section(
        "example.root-only", "ROOT-CANARY", execution_scope="root"
    )
    ctx.register_system_prompt_section("example.all", "ALL-CONTEXT")

    root = manager.render_system_prompt_sections({"delegation_depth": 0})
    delegated = manager.render_system_prompt_sections({"delegation_depth": 1})
    missing = manager.render_system_prompt_sections({})

    assert [section.id for section in root] == ["example.all", "example.root-only"]
    assert [section.id for section in delegated] == ["example.all"]
    assert [section.id for section in missing] == ["example.all"]


def test_prompt_registration_rejects_unknown_execution_scope():
    manager = PluginManager()
    with pytest.raises(ValueError, match="execution_scope"):
        _context(manager).register_system_prompt_section(
            "example.bad-scope", "content", execution_scope="nested"
        )


def test_constructor_time_child_context_cannot_render_root_prompt_section():
    agent = type("AgentUnderConstruction", (), {"session_id": "child"})()

    unbound_info = _plugin_session_info(agent)
    assert unbound_info["execution_kind"] == "unknown"
    assert unbound_info["delegation_depth"] == -1
    assert unbound_info["delegation_role"] == "unknown"
    with delegated_child_context("child"):
        info = _plugin_session_info(agent)

    assert info["execution_kind"] == "delegated"
    assert info["delegation_depth"] == 1
    assert info["delegation_role"] == "leaf"


def test_child_prompt_identity_rejects_unstructured_host_role():
    agent = type(
        "MalformedRoleAgent",
        (),
        {"session_id": "child", "_delegate_depth": 1, "_delegate_role": ["leaf"]},
    )()

    info = _plugin_session_info(agent)

    assert info["execution_kind"] == "delegated"
    assert info["delegation_role"] == "leaf"


@pytest.mark.parametrize("raw_depth", [None, [], -1, True])
def test_malformed_prompt_depth_is_unknown_and_cannot_render_root_section(raw_depth):
    manager = PluginManager()
    _context(manager).register_system_prompt_section(
        "example.root-only", "ROOT-CANARY", execution_scope="root"
    )
    agent = type(
        "MalformedDepthAgent",
        (),
        {"session_id": "unknown", "_delegate_depth": raw_depth},
    )()

    info = _plugin_session_info(agent)

    assert info["execution_kind"] == "unknown"
    assert info["delegation_depth"] == -1
    assert info["delegation_role"] == "unknown"
    assert manager.render_system_prompt_sections(info) == []


@pytest.mark.parametrize("raw_depth", [65, 10**100])
def test_positive_prompt_depth_remains_delegated(raw_depth):
    agent = type(
        "DeepDelegatedAgent",
        (),
        {
            "session_id": "deep-child",
            "_delegate_depth": raw_depth,
            "_delegate_role": "orchestrator",
        },
    )()

    info = _plugin_session_info(agent)

    assert info["execution_kind"] == "delegated"
    assert info["delegation_depth"] == raw_depth
    assert info["delegation_role"] == "orchestrator"
