# Architecture Decision Records

## 2026-08-26: Broker Codex Computer Use through the signed Codex app-server daemon

Status: Accepted

Context:
Hermes publishes a checked-in, exact ten-tool Computer Use surface and applies
its own untrusted-tool approvals, exact-action grants, state digests,
single-writer lease, and result sanitation. The previous transport executed
the installed `SkyComputerUseClient` directly. On macOS, that gives the
Computer Use service an unsigned Hermes/Python ancestry instead of the signed
OpenAI host ancestry that owns its privacy grants. It can therefore trigger
repeated Screen Recording prompts even when the OpenAI application has already
been granted access.

The signed Codex CLI bundled in ChatGPT 0.149 exposes a maintained app-server
daemon with a Unix-socket WebSocket transport. OpenAI's daemon lifecycle
intentionally uses the standalone binary under
`CODEX_HOME/packages/standalone/current`, so the launcher CLI and running
app-server can have different versions. OpenAI documents both the daemon and
app-server protocol as experimental and guarantees generated schemas only for
the Codex version that generated them. The generated 0.149 v2 protocol has
the three model-free calls needed here: `mcpServerStatus/list`,
`mcpServer/tool/call`, and `thread/delete`. None of these operations requires
`turn/start`, and Hermes must never invoke a Codex model to execute a Computer
Use call.

Live protocol proof against signed managed `0.149.0-alpha.4.3` exposed two
important lifecycle details. First, MCP startup emits roughly thirty status
notifications before returning the full catalog. Second, threads created with
`ephemeral: true` reject `thread/delete`, and closing their client connection
leaves their MCP child stack until the daemon's 30-minute idle unload. A
temporary thread created with `ephemeral: false` can be hard-deleted: deletion
returns success, removes it from list/read, leaves no rollout file, and tears
down the newly created MCP children.

Decision:
- Keep Claude Opus as the sole orchestrator. Route each granted Computer Use
  operation through a new `CodexCUABroker.call(...)` boundary that hides
  daemon discovery, trust validation, one fresh socket connection, one
  temporary deletable control thread, exact catalog attestation, the tool
  call, and mandatory thread deletion.
- Resolve only `/Applications/ChatGPT.app/Contents/Resources/codex` as the
  signed lifecycle controller. Before a daemon start, verify the ChatGPT
  bundle and CLI designated requirements for OpenAI team `2DC432GLL2`, reject
  symbolic links, unsafe ownership or mode on every embedded path component,
  and recheck the same vnode identities immediately before execution.
- Treat the OS account's `CODEX_HOME/app-server-control` directory and Unix
  socket as a security boundary. Reject links, wrong ownership, permissive
  modes, non-socket endpoints, and version/catalog drift. Parse the complete
  machine-readable daemon-version response, require its launcher version to
  match the embedded controller, resolve its exact official `managedCodexPath`
  alias to the versioned standalone release, verify that real binary's OpenAI
  signature and self-reported version, and snapshot both the immutable release
  chain and official aliases. Connect the raw Unix socket, read macOS
  `LOCAL_PEERPID`, require its `proc_pidpath` to equal that verified managed
  realpath, and bind that live PID through Security.framework to the exact
  OpenAI designated requirement before the WebSocket handshake. The
  `codesign -R` CLI receives its required leading `=` wrapper; the equivalent
  Security.framework parser string does not, because Apple rejects that CLI
  wrapper with `errSecCSReqInvalid`.
  Start/readiness work is coordinated by an account-global lock and a minimal
  environment. The shared daemon outlives Hermes panes and is not stopped
  during MCP shutdown.
- Do not infer compatibility from semver ordering. Accept only explicitly
  tested protocol versions, initially `0.149.0-alpha.4.3`, and require
  `managedCodexVersion == appServerVersion == initialize userAgent version`.
  Launcher/managed equality is not required. The observed signed 0.147 daemon
  remains rejected: its full status row omits `pluginId`, changes
  `serverInfo`, and changes the catalog contract. Operators
  move the managed runtime through OpenAI's exact-release installer and daemon
  restart; Hermes never substitutes or downloads a binary itself.
- For every call, connect directly to the Unix socket with WebSocket framing,
  initialize, start a temporary persisted thread (`ephemeral: false`), require
  its nonempty rollout path, request the complete MCP status
  catalog, and attest the exact CUA plugin id and immutable ten-tool digest
  before sending `mcpServer/tool/call`. Once `thread/start` returns its id,
  delete the thread in `finally` on success, server error, timeout, or
  cancellation. Do not accept the ephemeral-thread `-32600` deletion response
  as cleanup success for this persisted contract. If creation itself loses its
  response before the id is known, close the fresh connection and send no tool
  call; the protocol exposes no safe correlated thread identity to delete.
- ChatGPT Codex `0.149.0-alpha.4.3`'s generated
  `ListMcpServerStatusResponse` has no `runtimeStatus` field. Require its real
  full thread-scoped shape instead: one exact server/plugin row, explicit
  `authStatus: unsupported`, exact `Computer Use` server identity/version,
  zero resources/templates, all ten complete App-Server-decorated Tool
  contracts and annotations, and no unknown row/tool fields. The subsequent
  live `mcpServer/tool/call` is the operational proof. Unknown future fields
  fail closed rather than disappearing from the catalog digest.
- Bound catalog pages and rows and share one absolute deadline across binary
  attestation, daemon readiness, initialization, thread creation, catalog, and
  tool dispatch. Ordinary model-runtime event queues remain bounded and
  terminal on overflow. The model-free broker explicitly discards only
  syntactically valid notifications after JSON-RPC envelope/method validation,
  because it has no event consumer. The broker promptly rejects every
  server-initiated request because Hermes's own grants and high-impact
  confirmation are the only approval authority.
- Keep all existing Hermes gates before broker work: trust/confirmation,
  exact grant/state checks, account-global single-writer flock, and circuit
  breaker. Reuse the existing MCP result renderer so text, images, audio,
  resources, structured content, metadata, errors, and trusted state hashes
  retain one contract. Validate App Server results through the same strict MCP
  SDK `CallToolResult` content union before rendering.
- Never fall back to the direct Sky executable. Never use `codex mcp-server`;
  it is a deprecated, model-mediated route. Do not retry after a frame may
  have been accepted. The current broker performs no automatic retry. A future
  bounded retry may be added only for an authenticated, explicit
  pre-execution overload response while the existing lease remains held.
- Represent the host config with reserved transport type
  `codex_app_server`. The entry is commandless and accepts no args, env, or
  URL. Only the shipped exact catalog entry may claim this transport and the
  `openai-codex-cua` identity.

Alternatives considered:
- Keep the direct Sky launcher. Rejected because it cannot supply the signed
  OpenAI process ancestry that macOS privacy authorization expects.
- Spawn `codex app-server proxy` for every call and keep stdio semantics.
  Rejected because it adds one child/process pipe per call, expands lifecycle
  and retry ambiguity, and bypasses the maintained shared-daemon boundary.
- Expose app-server/thread orchestration directly from `mcp_tool.py`.
  Rejected because callers would need to understand version checks, socket
  trust, temporary-thread hard deletion, catalog attestation, and protocol
  ambiguity.
  The broker is a deeper and safer boundary.

Consequences:
- Hermes gains Codex's supported macOS Computer Use host ancestry without
  changing its orchestrator or delegating the decision to a Codex model.
- A Computer Use call fails closed if the ChatGPT/Codex installation, daemon,
  connected peer, socket, protocol version, plugin identity, result envelope,
  or tool catalog drifts.
- A newer signed launcher may reuse an older signed managed daemon only when
  the managed/app-server protocol version is on the explicit compatibility
  allowlist; currently this means exact 0.149.0-alpha.4.3 only.
- The first granted call may pay daemon-readiness latency; later calls reuse
  one account-global daemon but still receive isolated connections and
  temporary persisted threads that are hard-deleted before disconnect.
- A hard process kill can occur after `thread/start` returns but before the
  `finally` block deletes its thread. The current protocol does not expose an
  atomic broker marker: `serviceName` is not retained, and no-turn control
  threads are absent from both `thread/list` and `thread/loaded/list`. Hermes
  therefore does not scan or delete user threads by a weak cwd/name heuristic.
  Codex's 30-minute idle unload bounds the in-memory MCP stack in this crash
  case; an operator daemon restart is the only immediate recovery. A cleanup
  RPC is not retried after an ambiguous accepted frame because the generated
  protocol does not declare `thread/delete` idempotent.
- The implementation is macOS-only and intentionally has no compatibility
  fallback to an unsigned or model-mediated launcher.
- The cooperative OS account and the Hermes code/configuration it loads are
  trusted. Code already executing maliciously as that account can alter this
  Python process and its files directly and is outside this boundary. The
  vnode/codesign checks mitigate installation drift and nonconcurrent
  replacement; live peer attestation mitigates stale or replaced endpoints.
  They do not claim to eliminate the pathname verify-to-exec race after the
  account itself is compromised; macOS provides no supported `fexecve` or
  atomic Security.framework validate-and-launch primitive for this CLI.

## 2026-07-13: Scope plugin manager state by Hermes home/profile (keyed cache)

Status: Accepted

Context:
Hermes supports multiple profiles via different Hermes home directories.
Homes are switched two ways in a running process: the `HERMES_HOME`
environment variable (single-profile CLI/gateway processes), and the
context-local `set_hermes_home_override()` (`hermes_constants.py`), which
the multiplexed gateway worker (`gateway/run.py`'s `_profile_scope`) and
subagent/embedded callers use to serve several profiles from one
long-lived process. The override is a `ContextVar` and deliberately does
**not** mutate `os.environ`, since that would leak one profile's home
into every other concurrent task in the same process.

The plugin manager was a process-global single-slot singleton
(`_plugin_manager`). User-installed plugins are discovered from
`get_hermes_home() / "plugins"`, and context-engine plugins (e.g.
`hermes-lcm`) capture profile-scoped state — such as the LCM database
path — at registration time. A single-slot cache meant:

1. Switching homes via `set_hermes_home_override()` was invisible to a
   naive "did `HERMES_HOME` change" check, so the singleton silently kept
   serving the first profile's manager to every other profile in the
   process.
2. Even when a fresh `PluginManager` *was* created for a new home, plugin
   modules are imported into `sys.modules` as `hermes_plugins.<slug>` by
   `_load_directory_module`, and only that top-level module was ever
   replaced. A same-slug plugin's *relative* imports
   (`from . import state`) are cached separately under
   `hermes_plugins.<slug>.<submodule>`, and Python's import machinery
   resolves those from `sys.modules` first — so a profile switch could
   silently keep serving a previous profile's already-imported submodule
   code/state instead of re-executing the new profile's plugin.

Decision:
- Replace the single-slot singleton with a cache keyed on the *resolved*
  Hermes home path (`_plugin_managers_by_home: Dict[Path, PluginManager]`).
  `get_plugin_manager()` resolves the current home via `get_hermes_home()`
  (which itself already consults `get_hermes_home_override()` before
  `os.environ`), so both the env-var and context-local override paths are
  covered uniformly.
- `_plugin_manager` (the old single-slot name) is kept as a thin "last
  manager returned" pointer purely for backward compatibility with
  existing test code that does
  `monkeypatch.setattr(plugins_mod, "_plugin_manager", some_manager)`.
  When that name is monkeypatched to a manager the keyed cache doesn't
  know about, `get_plugin_manager()` treats it as an explicit injection
  and adopts it into the cache under the *current* resolved home, rather
  than discarding it.
- Both `PluginManager._load_directory_module` (initial/`force=True`
  reload within the same home) and the shared `_clear_plugin_submodules`
  helper (profile switch / test teardown) evict `sys.modules[module_name]`
  **and every name prefixed with `module_name + "."`** before a plugin
  slug is (re-)imported, so relative-import submodules can never survive
  a reload or a home switch.
- Test isolation (`tests/conftest.py`'s `_hermetic_environment` fixture)
  calls a new `_reset_plugin_managers_for_tests()` helper that drops the
  entire keyed cache and purges every plugin submodule from `sys.modules`
  between tests, instead of only resetting the single-slot pointer.

Consequences:
- Per-profile LCM instances (and any other context-engine plugin) use
  their own `{home}/lcm.db` regardless of whether the profile switch went
  through `HERMES_HOME` or `set_hermes_home_override()`.
- Plugin discovery remains cached within a profile for normal
  performance, and re-entering a previously-seen profile reuses its
  cached manager instead of rebuilding from scratch.
- Sequential *and* interleaved profile switching — in tests, the gateway
  multiplexer worker, or embedded callers using the context-local
  override — no longer leaks context-engine state, plugin module state,
  or stale relative-import submodules across profiles.
- Regression coverage exercises the real production path
  (`set_hermes_home_override()`) rather than only the env-var path, and
  includes a dedicated relative-import leak test.
