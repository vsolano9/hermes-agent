---
title: Public Subagent Lifecycle API
sidebar_label: Subagent lifecycle API
---

# Public Subagent Lifecycle API

Native plugins can launch and supervise fresh Hermes child sessions through a
host-owned, asynchronous API. Obtain the API from `PluginContext`; do not
construct lifecycle services or import delegation internals.

```python
from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLaunchRequestV2

def launch_review(args, *, invocation):
    request = SubagentLaunchRequestV2(
        api_contract_version=2,
        base=SubagentLaunchRequest(
            goal="Review this change and report regressions.",
            correlation_id="review-42",
            model="my-worker-model",
        ),
        provider="my-provider",
        reasoning_effort="high",
        toolset_mode="exact",
        exact_toolsets=(),
    )
    return invocation.subagents.launch(request).to_dict()

def register(ctx):
    ctx.register_tool("launch_review", "reviewer", REVIEW_SCHEMA, launch_review)
```

A keyword-only parameter named `invocation` opts a tool handler into the
invocation contract. A positional-or-keyword parameter does not opt in unless
registration passes `inject_invocation=True`; legacy handlers and their exact
keyword payload remain unchanged. `invocation.operation_id` is a bounded,
host-minted identifier for one handler execution. It changes between turns and
is audit metadata, not authority. Legacy `task_id` remains the session ID.

`ctx.subagent_lifecycle` and `invocation.subagents` are host-minted facades.
They expose lifecycle methods, not agents, credentials, parent objects, or an
authority constructor. Each operation is checked against the active plugin,
canonical profile, plugin-manager scope, and session. A handle retained by the
same plugin can be used in a later turn of the same session even though the
operation ID changed. Cross-plugin, cross-profile, cross-manager, cross-session,
forged, expired, and nested-descendant access fails closed.

## Requests and toolsets

`SubagentLaunchRequest` is the v1 compatibility request. Its
`allowed_toolsets=None` and `allowed_toolsets=()` forms both retain the legacy
inherit behavior.

`SubagentLaunchRequestV2(api_contract_version=2, ...)` makes toolset intent
explicit:

- `toolset_mode="inherit"` inherits the parent's enabled toolsets.
- `toolset_mode="exact"` resolves exactly `exact_toolsets` as a safe subset of
  the parent's effective toolsets.
- Exact-empty means zero Hermes model-visible tools. It does not inherit MCP
  tools, re-add delegation for an orchestrator role, or mutate parent toolsets.

Unknown, parent-broadening, or mutation-unsafe exact requests are rejected
before child construction. `blocked_tools`, per-launch timeouts, and every
non-`None` `working_directory` are unsupported and fail before admission.

An explicit v2 provider/model route is resolved inside the launch profile by
Hermes's native provider catalog and credential path. Only public identifiers
cross the plugin boundary. Omitted reasoning preserves configured behavior;
explicit `False`/`"none"` disables reasoning; invalid values are rejected.
Credentials, base URLs, environment values, commands, and ACP arguments are
never added to public requests, handles, diagnostics, or audit metadata.

Exact-empty is necessary but not sufficient for a coordinator's native
read-only route. Hermes also verifies the instantiated child has zero model
tools and an allowlisted native transport with no independent mutation
channel. This transport assessment is host-private. `copilot-acp` is not
eligible for this read-only route because ACP independently exposes filesystem
mutation methods; it fails before the process starts.

## Handles, controls, and completion

`SubagentHandle` uses serialized handle contract version 1. The lifecycle API
contract is independently versioned as v2. Persist `handle.to_dict()` and load
it with `SubagentHandle.from_dict()`; a handle does not survive a host restart.

The v1 methods remain supported: `status`, `wait`, `cancel`, `result`, and
`reconnect`. The v2 control surface adds:

- `list()` — statuses for lifecycle-launched direct children only;
- `steer(handle, message)` — one queued correction, or a stable missed,
  terminal, unsupported, or unknown disposition;
- `stop(handle, reason=...)` — cooperative cancellation;
- `collect(handle)` — a not-ready receipt or an immutable terminal receipt.

Ready collection is idempotent: repeated calls return the same stable
`event_id`, `collected_at`, terminal state, and hash-bound immutable result.
Arbitrary owner-authorized result content is bounded and protected data; only
core-generated diagnostics and metadata are allowlisted and sanitized.

The stable states are `PENDING`, `STARTING`, `RUNNING`, `SUCCEEDED`, `FAILED`,
`INTERRUPTED`, `CANCEL_REQUESTED`, `CANCELLED`, and `UNKNOWN`. Native in-flight
work is process-local. After restart, prior handles return `UNKNOWN` or
`RECONNECT_UNAVAILABLE`; Hermes never starts a replacement child.

## Shared admission

Lifecycle launches, background `delegate_task`, and direct cron dispatch share
one atomic host capacity counter plus the native pause and depth gates.
Lifecycle returns a bounded rejection such as `CAPACITY_REACHED` before child
construction. Legacy background `delegate_task` preserves its synchronous
fallback only for capacity denial, including its consolidated batch result.
Direct cron runs preserve their existing result contract and release the same
shared lease exactly once.

Terminal lifecycle records are retained in-process for one hour. Plugin unload
revokes its facades and waits for already-admitted mutating operations; no new
launch or control operation begins after revocation starts.
