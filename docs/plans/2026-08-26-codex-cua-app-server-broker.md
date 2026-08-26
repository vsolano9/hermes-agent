# Codex CUA App-Server Broker Implementation Plan

> **For the executor:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Route Hermes's exact ten Codex Computer Use tools through the signed, model-free Codex app-server daemon while preserving every existing Hermes approval, grant, serialization, and result contract.

**Architecture:** `agent/transports/codex_app_server.py` gains a connection abstraction shared by the existing stdio client and a Unix-domain WebSocket connection. A new `agent/transports/codex_cua_broker.py` owns signed binary resolution, daemon/socket readiness, per-call temporary persisted control threads, exact app-server MCP attestation, direct tool calls, and hard-delete cleanup. `tools/mcp_tool.py` selects the broker only for the reserved host-pinned transport after all existing gates and renders its response through the same MCP sanitation path.

**Tech Stack:** Python 3.11+, JSON-RPC 2.0, `websockets==15.0.1`, Unix domain sockets, macOS `codesign`, `fcntl.flock`, pytest through `scripts/run_tests.sh`.

---

## Execution constraints

- Work only in `feat/codex-app-server-broker`; leave changes uncommitted for the main agent.
- Do not mutate AgentGrid, Chrome, TCC, live Hermes/Codex config, daemon state, or active processes during implementation tests.
- Preserve the immutable exact-ten schema, `openai-codex-cua` identity, trust gate, exact grant/state digest, account-global single-writer lease, circuit breaker, and result rendering.
- A call may create no daemon/thread work until those gates pass. No `turn/start`, model selection, direct Sky execution, or `codex mcp-server` fallback is permitted.
- Use actual generated app-server protocol shapes in fixtures. Do not assert by reading production source or snapshotting mutable catalogs.
- The shared daemon is account-global and is never stopped by Hermes shutdown/reload.
- The cooperative OS account and loaded Hermes code/configuration are trusted.
  Signature/vnode/peer checks defend installation drift and stale or replaced
  endpoints; they do not claim resistance to arbitrary code already executing
  as that same account, which could alter Hermes directly.

## Slice 1: Connection-neutral JSON-RPC client over stdio and UDS WebSocket

**Behavior seam:** `CodexAppServerClient` speaks the same request/notification/error contract over either a subprocess-backed stdio connection or a supplied connection implementation.

**Paths:**
- Modify `agent/transports/codex_app_server.py`
- Add `tests/agent/transports/test_codex_app_server_uds.py`
- Update existing `tests/agent/transports/test_codex_app_server_runtime.py` only where compatibility requires it

**RED:** Add a real local Unix-socket WebSocket fixture that proves initialize/request/notification/server-request/error/close behavior, immediate rejection policy, bounded event queues, and rejection of binary, oversized, malformed JSON, and malformed JSON-RPC frames. Run:

```bash
scripts/run_tests.sh tests/agent/transports/test_codex_app_server_uds.py -q
```

Expected RED: the client has no UDS WebSocket connection seam.

**GREEN:** Extract a small connection protocol with `send`, receive callback/iterator, liveness, diagnostics, and close. Keep subprocess spawning entirely inside the stdio implementation. Add a synchronous UDS WebSocket implementation using the pinned `websockets` dependency and preserve the existing client API.

**Proof:** Run the new UDS tests plus all existing app-server transport/session tests. Confirm stdio behavior is unchanged.

## Slice 2: Trusted signed Codex and daemon/socket readiness

**Behavior seam:** `CodexAppServerDaemonController.ensure_ready()` returns an attested socket endpoint only when the signed ChatGPT Codex installation, account-level control paths, and running version are exact.

**Paths:**
- Add `agent/transports/codex_cua_broker.py`
- Add `tests/agent/transports/test_codex_cua_broker_trust.py`

**RED:** Build filesystem fixtures for the expected ChatGPT path and `CODEX_HOME/app-server-control` path. Cover fixed-path enforcement, exact designated requirements, linked or writable components, same-size replacement with restored mtime, unsafe control-directory mode, wrong endpoint type/owner, version mismatch, and sanitized failures. Add an unexpected local Unix peer and prove rejection before WebSocket upgrade.

**GREEN:** Implement exact embedded-controller resolution, designated-requirement verification for team `2DC432GLL2`, full component vnode snapshots/recheck, minimal child environment, bounded daemon start/readiness, and full machine-readable version parsing. Resolve the official managed alias to its versioned standalone realpath, verify and snapshot that signed binary, and authenticate the connected socket against that managed identity with `LOCAL_PEERPID`, exact `proc_pidpath`, and Security.framework's live-code designated-requirement check. Treat launcher/managed skew as an official lifecycle property, but accept only explicitly proven protocol versions (initially exact `0.149.0-alpha.4.3`) with `managed == appServer == initialize`; never infer compatibility from semver. Keep the live-observed 0.147 daemon rejected. Never kill the daemon.

**Proof:** Run the trust suite repeatedly and verify it creates no live daemon or user-config mutation by injecting the command/connection seams.

## Slice 3: One model-free, deletable, attested broker call

**Behavior seam:** `CodexCUABroker.call(tool, arguments, timeout)` returns one MCP `CallToolResult`-shaped value after an exact catalog attestation and always attempts `thread/delete`.

**Paths:**
- Modify `agent/transports/codex_cua_broker.py`
- Add `tests/agent/transports/test_codex_cua_broker_call.py`

**RED:** Use protocol fixtures shaped like generated ChatGPT 0.149 schemas. Assert the exact sequence `initialize` → `thread/start {ephemeral:false}` → paginated `mcpServerStatus/list` → `mcpServer/tool/call` → `thread/delete`; assert no `turn/start`. Cover the normal MCP startup notification burst, exact plugin/server id and digest, catalog/version drift, pagination bounds/deadline, success, MCP error, generic RPC ambiguity, timeout, cancellation, transport loss, malformed MCP content, and cleanup failure.

**GREEN:** Implement a fresh UDS connection and temporary persisted thread per call, exact catalog/tool-schema digest attestation, one direct tool call, and hard deletion in `finally`. The broker discards validated notifications rather than accumulating an unread queue; ordinary model-runtime clients retain their bounded queue. Mark the call ambiguous once its request frame is accepted and prohibit retry thereafter. The generated protocol does not provide an authenticated pre-execution overload shape, so this implementation conservatively performs no automatic retry; any future bounded retry must first prove that explicit pre-execution signal while retaining the lease.

**Protocol evidence adjustment:** The generated
`0.149.0-alpha.4.3` `ListMcpServerStatusResponse` schema contains no
`runtimeStatus`. Attestation therefore requires the generated full row shape,
exact plugin id, `authStatus: unsupported`, zero resources/templates, exact
full Tool contracts/annotations, and rejection of unknown fields. The direct
tool call supplies operational proof; no nonexistent field is synthesized.

**Proof:** Run a complete broker call through a real local UDS WebSocket fixture and confirm the exact sequence, deterministic rejection of a server-initiated elicitation, strict SDK result validation, and deletion. Unit seams confirm known-result cleanup failures never mask the result and post-dispatch uncertainty is never presented as retryable. If `thread/start` itself loses the uncorrelated id, the isolated connection closes and no tool call is sent.

## Slice 4: Reserved catalog transport and host-pinned dispatch

**Behavior seam:** The shipped `codex-computer-use` catalog entry produces a commandless `codex_app_server` config that alone can publish the exact ten tools and dispatch through the broker after Hermes gates.

**Paths:**
- Modify `optional-mcps/codex-computer-use/manifest.yaml`
- Remove or retire `optional-mcps/codex-computer-use/launcher.py` and its launcher-only tests if no longer referenced
- Modify `hermes_cli/mcp_catalog.py`
- Modify `tools/mcp_pinned_surfaces.py`
- Modify `tools/mcp_tool.py`
- Modify `tests/hermes_cli/test_codex_computer_use_catalog.py`
- Modify `tests/hermes_cli/test_codex_computer_use_launcher.py` or replace it with broker trust coverage
- Modify `tests/tools/test_mcp_pinned_lazy.py`

**RED:** Assert the manifest/config has reserved type `codex_app_server` and no command/args/env/url; ordinary aliases and copied digest pairs cannot claim it; cold registration publishes ten tools with zero broker/daemon activity; denial, missing grant, writer contention, and open circuit breaker cause zero broker work; eight processes serialize aliases to one call; shutdown/reload retains exact ten static definitions and never stops the shared daemon.

**GREEN:** Teach catalog parsing/building the reserved transport with exact-entry checks. Replace launcher identity checks with shipped-entry transport identity. Dispatch pinned handlers to the broker without creating `MCPServerTask`. Extract/reuse the MCP result renderer so broker and normal MCP calls share images, structured content, errors, metadata, state hashes, and grants.

**Proof:** Run pinned, grant, catalog, plugin, delegation, and MCP tool suites. Verify direct Sky and `codex mcp-server` strings are absent from reachable broker routing.

## Slice 5: Concurrency, lifecycle, and regression proof

**Behavior seam:** Across process aliases and reloads, exactly one granted CUA call owns the account-global lease while the shared daemon remains stable and every invocation is isolated and cleaned up.

**Paths:**
- Add focused multiprocess/integration coverage under `tests/tools/` and `tests/agent/transports/`
- Update concise operator/developer documentation only if the actual shipped behavior needs it

**RED/GREEN cases:**
- One serialized broker call across aliases under the existing account-global flock.
- Stale-flock recovery without granting an ordinary alias the reserved identity.
- Success/error/timeout/cancel always hard-delete the temporary persisted thread.
- Reload/shutdown publish exactly ten tools with zero eager broker work.
- Ordinary aliases never gain the reserved identity or shared capability lock.
- No replay after an ambiguous accepted frame; there is no automatic retry.

Hermetic tests inject daemon readiness and never start, stop, or replace a live
daemon. The managed `daemon start` command is idempotent and executes while the
existing call lease is held; Codex owns the single account-global daemon. The
main agent owns post-integration live acceptance of the signed peer/version and
daemon reuse on the user's installed build.

**Validation commands:**

```bash
scripts/run_tests.sh tests/agent/transports/ -q
scripts/run_tests.sh tests/tools/test_mcp_pinned_lazy.py tests/tools/test_mcp_routine_grants.py -q
scripts/run_tests.sh tests/hermes_cli/test_codex_computer_use_catalog.py tests/hermes_cli/test_mcp_catalog.py -q
scripts/run_tests.sh tests -q --collect-only
```

Then run every MCP-named test file through `scripts/run_tests.sh`, relevant delegation/catalog/plugin suites, and:

```bash
python -m py_compile agent/transports/codex_app_server.py agent/transports/codex_cua_broker.py tools/mcp_pinned_surfaces.py tools/mcp_tool.py hermes_cli/mcp_catalog.py
python -m compileall -q agent/transports tools hermes_cli
git diff --check
git status --short
```

Review the complete diff, scan changed files for credentials/tokens/private paths, check for test residue and background processes, and obtain an independent hostile review. The main agent owns commit, push, rollout, and live acceptance.

## Recovery and stop condition

The safe recovery point after each slice is the prior green test set. Revert only the isolated slice if its public seam proves wrong; never weaken identity or replay checks to make a test pass. Stop when the complete uncommitted diff passes focused and broad validation, hostile review has no unresolved high/medium issue, no task residue remains, and the main agent has exact changed-file and command evidence for integration.
