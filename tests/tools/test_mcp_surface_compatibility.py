"""Fail-closed protocol-surface compatibility for pinned MCP servers."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from tools import mcp_tool
from tools.registry import ToolRegistry


def _tool(index: int):
    read_only = index < 2
    return SimpleNamespace(
        name=f"tool_{index}",
        title=f"Tool {index}",
        description=f"Description {index}",
        inputSchema={
            "type": "object",
            "properties": {"app": {"type": "string"}},
            "required": ["app"],
            "additionalProperties": False,
        },
        annotations={
            "destructiveHint": False,
            "idempotentHint": read_only,
            "openWorldHint": False,
            "readOnlyHint": read_only,
        },
        outputSchema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        execution=None,
        icons=None,
    )


def _installed_shape():
    return [_tool(index) for index in range(10)]


def _config(tools):
    caps = _caps()
    return {
        "compatibility": {
            "tools_sha256": mcp_tool._mcp_surface_sha256(tools),
            "capabilities_sha256": mcp_tool._mcp_capabilities_sha256(caps),
            "tool_count": 10,
            "tools_only": True,
        }
    }


def _caps():
    return SimpleNamespace(
        completions=None,
        experimental=None,
        extensions=None,
        logging=None,
        prompts=None,
        resources=None,
        tasks=None,
        tools=SimpleNamespace(listChanged=False),
    )


def test_exact_surface_is_accepted():
    tools = _installed_shape()
    mcp_tool._validate_mcp_surface("codex", tools, _caps(), _config(tools))


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "schema", "description", "title", "output", "annotation"]
)
def test_surface_mutations_fail_closed(mutation):
    expected = _installed_shape()
    actual = deepcopy(expected)
    if mutation == "missing":
        actual.pop()
    elif mutation == "extra":
        actual.append(_tool(10))
    elif mutation == "schema":
        actual[2].inputSchema["required"] = ["app", "element_index"]
    elif mutation == "description":
        actual[2].description = "Injected description"
    elif mutation == "title":
        actual[2].title = "Changed title"
    elif mutation == "output":
        actual[2].outputSchema["required"] = ["ok"]
    else:
        actual[2].annotations["openWorldHint"] = True

    with pytest.raises(ValueError, match="compatibility check failed"):
        mcp_tool._validate_mcp_surface("codex", actual, _caps(), _config(expected))


def test_duplicate_names_are_rejected_even_when_digest_matches():
    tools = _installed_shape()
    tools[-1].name = tools[0].name

    with pytest.raises(ValueError, match="duplicate tool name"):
        mcp_tool._validate_mcp_surface("codex", tools, _caps(), _config(tools))


def test_resources_or_prompts_capability_fails_tools_only_contract():
    tools = _installed_shape()
    with pytest.raises(ValueError, match="tools-only"):
        mcp_tool._validate_mcp_surface(
            "codex",
            tools,
            SimpleNamespace(resources=object(), prompts=None),
            _config(tools),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tools", None),
        ("tools", SimpleNamespace(listChanged=True)),
        ("logging", SimpleNamespace()),
        ("completions", SimpleNamespace()),
        ("experimental", {"injected": True}),
        ("extensions", {"other": True}),
        ("tasks", SimpleNamespace()),
        ("other", "drift"),
    ],
)
def test_initialize_capability_mutations_fail_closed(field, value):
    tools = _installed_shape()
    expected_caps = _caps()
    actual_caps = deepcopy(expected_caps)
    setattr(actual_caps, field, value)

    with pytest.raises(ValueError, match="capability contract changed"):
        mcp_tool._validate_mcp_surface(
            "codex", tools, actual_caps, _config(tools)
        )


def test_refresh_schema_drift_quarantines_every_published_handler():
    expected = _installed_shape()
    drifted = deepcopy(expected)
    drifted[4].description = "Untrusted replacement instructions"
    registry = ToolRegistry()
    server = mcp_tool.MCPServerTask("codex")
    server._config = _config(expected)
    server.initialize_result = SimpleNamespace(capabilities=_caps())
    server._tools = expected
    server._registered_tool_names = []

    for tool in expected:
        name = mcp_tool.mcp_prefixed_tool_name("codex", tool.name)
        registry.register(
            name=name,
            toolset="mcp-codex",
            schema={},
            handler=lambda **_kwargs: None,
            check_fn=lambda: True,
            is_async=False,
            description="",
        )
        server._registered_tool_names.append(name)

    server.session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=drifted))
    )

    with patch("tools.registry.registry", registry):
        with pytest.raises(ValueError, match="compatibility check failed"):
            asyncio.run(server._refresh_tools())

    assert server._registered_tool_names == []
    assert server._tools == []
    assert not any(
        registry.get_toolset_for_tool(name) == "mcp-codex"
        for name in registry.get_all_tool_names()
    )


def test_clean_reconnect_revalidates_before_old_handlers_remain_callable():
    expected = _installed_shape()
    drifted = deepcopy(expected)
    drifted[3].inputSchema["properties"]["injected"] = {"type": "string"}
    registry = ToolRegistry()
    server = mcp_tool.MCPServerTask("codex")
    server._config = _config(expected)
    server.initialize_result = SimpleNamespace(capabilities=_caps())
    server._tools = expected
    server._ready.set()
    for tool in expected:
        name = mcp_tool.mcp_prefixed_tool_name("codex", tool.name)
        registry.register(
            name=name,
            toolset="mcp-codex",
            schema={},
            handler=lambda **_kwargs: None,
            check_fn=lambda: True,
            is_async=False,
            description="",
        )
        server._registered_tool_names.append(name)
    server.session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=drifted))
    )

    with patch("tools.registry.registry", registry):
        with pytest.raises(ValueError, match="compatibility check failed"):
            asyncio.run(server._discover_tools())

    assert server._tools == []
    assert server._registered_tool_names == []
    assert not server._ready.is_set()
    assert server._error is not None
    assert not any(
        registry.get_toolset_for_tool(name) == "mcp-codex"
        for name in registry.get_all_tool_names()
    )
