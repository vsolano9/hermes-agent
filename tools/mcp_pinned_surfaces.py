"""Immutable, host-owned MCP contracts that are safe to advertise pre-connect.

Only exact compatibility digest pairs select a surface.  Operator manifests
cannot supply descriptions or schemas, and user-writable schema caches are not
consulted.  Fresh objects are returned so live-server state cannot mutate the
checked-in authority.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import NamedTuple


CODEX_CUA_TOOLS_SHA256 = "dd485a140f5fbebe14147fb3ee2ed3914618b3484964efe02262b2479b322f1d"
CODEX_CUA_CAPABILITIES_SHA256 = "52aa21370a62916d63adb5718fa1be519ec0fe4390136bf36e701be54e5582a5"
CODEX_CUA_TOOL_NAMES = (
    "list_apps", "get_app_state", "click", "perform_secondary_action",
    "set_value", "select_text", "scroll", "drag", "press_key", "type_text",
)


class PinnedSurface(NamedTuple):
    tools: list[SimpleNamespace]
    capabilities: SimpleNamespace


def _annotations(read_only: bool) -> dict:
    return {
        "destructiveHint": False,
        "idempotentHint": read_only,
        "openWorldHint": False,
        "readOnlyHint": read_only,
    }


_APP = {"description": "App name, full app path, or unambiguous bundle identifier", "type": "string"}
_ELEMENT = {"description": "Element identifier", "type": "string"}

_CODEX_CUA_TOOLS = (
    {
        "name": "list_apps",
        "description": "List the apps on this computer. Returns the set of apps that are currently running, as well as any that have been used in the last 14 days, including details on usage frequency",
        "inputSchema": {"additionalProperties": False, "properties": {}, "type": "object"},
        "annotations": _annotations(True),
    },
    {
        "name": "get_app_state",
        "description": "Start an app use session if needed, then get the state of the app's key window and return a screenshot and accessibility tree. This must be called once per assistant turn before interacting with the app",
        "inputSchema": {"additionalProperties": False, "properties": {"app": _APP}, "required": ["app"], "type": "object"},
        "annotations": _annotations(True),
    },
    {
        "name": "click",
        "description": "Click an element by index or pixel coordinates from screenshot",
        "inputSchema": {"additionalProperties": False, "properties": {
            "app": _APP,
            "click_count": {"description": "Number of clicks. Defaults to 1", "type": "integer"},
            "element_index": {"description": "Element index to click", "type": "string"},
            "mouse_button": {"description": "Mouse button to click. Defaults to left.", "enum": ["left", "right", "middle"], "type": "string"},
            "x": {"description": "X coordinate in screenshot pixel coordinates", "type": "number"},
            "y": {"description": "Y coordinate in screenshot pixel coordinates", "type": "number"},
        }, "required": ["app"], "type": "object"},
        "annotations": _annotations(False),
    },
    {
        "name": "perform_secondary_action",
        "description": "Invoke a secondary accessibility action exposed by an element",
        "inputSchema": {"additionalProperties": False, "properties": {
            "action": {"description": "Secondary accessibility action name", "type": "string"},
            "app": _APP, "element_index": _ELEMENT,
        }, "required": ["app", "element_index", "action"], "type": "object"},
        "annotations": _annotations(False),
    },
    {
        "name": "set_value",
        "description": "Set the value of a settable accessibility element",
        "inputSchema": {"additionalProperties": False, "properties": {
            "app": _APP, "element_index": _ELEMENT,
            "value": {"description": "Value to assign", "type": "string"},
        }, "required": ["app", "element_index", "value"], "type": "object"},
        "annotations": _annotations(False),
    },
    {
        "name": "select_text",
        "description": "Select text inside a text element, or place the text cursor before or after it. Provide text exactly as it appears in the accessibility tree, including any Markdown formatting. If the text is not unique, provide surrounding prefix or suffix text to disambiguate it.",
        "inputSchema": {"additionalProperties": False, "properties": {
            "app": {"description": "App name or bundle identifier", "type": "string"},
            "element_index": {"description": "Text element identifier", "type": "string"},
            "prefix": {"description": "Optional text immediately before the target, used to disambiguate repeated matches", "type": "string"},
            "selection": {"description": "Whether to select the text or place the cursor before or after it. Defaults to text.", "enum": ["text", "cursor_before", "cursor_after"], "type": "string"},
            "suffix": {"description": "Optional text immediately after the target, used to disambiguate repeated matches", "type": "string"},
            "text": {"description": "Target text as shown in the accessibility tree", "type": "string"},
        }, "required": ["app", "element_index", "text"], "type": "object"},
        "annotations": _annotations(False),
    },
    {
        "name": "scroll",
        "description": "Scroll an element in a direction by a number of pages",
        "inputSchema": {"additionalProperties": False, "properties": {
            "app": _APP,
            "direction": {"description": "Scroll direction: up, down, left, or right", "type": "string"},
            "element_index": _ELEMENT,
            "pages": {"description": "Number of pages to scroll. Fractional values are supported. Defaults to 1", "type": "number"},
        }, "required": ["app", "element_index", "direction"], "type": "object"},
        "annotations": _annotations(False),
    },
    {
        "name": "drag",
        "description": "Drag from one point to another using pixel coordinates",
        "inputSchema": {"additionalProperties": False, "properties": {
            "app": _APP,
            "from_x": {"description": "Start X coordinate", "type": "number"},
            "from_y": {"description": "Start Y coordinate", "type": "number"},
            "to_x": {"description": "End X coordinate", "type": "number"},
            "to_y": {"description": "End Y coordinate", "type": "number"},
        }, "required": ["app", "from_x", "from_y", "to_x", "to_y"], "type": "object"},
        "annotations": _annotations(False),
    },
    {
        "name": "press_key",
        "description": "Press a key or key-combination on the keyboard, including modifier and navigation keys.\n  - This supports xdotool's `key` syntax.\n  - Examples: \"a\", \"Return\", \"Tab\", \"super+c\", \"Up\", \"KP_0\" (for the numpad 0 key).",
        "inputSchema": {"additionalProperties": False, "properties": {
            "app": _APP, "key": {"description": "Key or key combination to press", "type": "string"},
        }, "required": ["app", "key"], "type": "object"},
        "annotations": _annotations(False),
    },
    {
        "name": "type_text",
        "description": "Type literal text using keyboard input",
        "inputSchema": {"additionalProperties": False, "properties": {
            "app": _APP, "text": {"description": "Literal text to type", "type": "string"},
        }, "required": ["app", "text"], "type": "object"},
        "annotations": _annotations(False),
    },
)


def get_surface(tools_sha256: object, capabilities_sha256: object) -> PinnedSurface | None:
    """Resolve only an exact host-known fingerprint pair."""

    if (tools_sha256, capabilities_sha256) != (
        CODEX_CUA_TOOLS_SHA256, CODEX_CUA_CAPABILITIES_SHA256,
    ):
        return None
    tools = []
    for template in _CODEX_CUA_TOOLS:
        raw = deepcopy(template)
        raw.setdefault("title", None)
        raw.setdefault("outputSchema", None)
        raw.setdefault("execution", None)
        raw.setdefault("icons", None)
        tools.append(SimpleNamespace(**raw))
    capabilities = SimpleNamespace(
        completions=None, experimental=None, extensions=None, logging=None,
        prompts=None, resources=None, tasks=None,
        tools=SimpleNamespace(listChanged=False),
    )
    return PinnedSurface(tools, capabilities)
