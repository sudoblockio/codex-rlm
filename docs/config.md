# Configuration

For basic configuration instructions, see [this documentation](https://developers.openai.com/codex/config-basic).

For advanced configuration instructions, see [this documentation](https://developers.openai.com/codex/config-advanced).

For a full configuration reference, see [this documentation](https://developers.openai.com/codex/config-reference).

## Connecting to MCP servers

Codex can connect to MCP servers configured in `~/.codex/config.toml`. See the configuration reference for the latest MCP server options:

- https://developers.openai.com/codex/config-reference

## Notify

Codex can run a notification hook when the agent finishes a turn. See the configuration reference for the latest notification settings:

- https://developers.openai.com/codex/config-reference

## RLM runtime

RLM (Recursive Language Model) provides tools for processing large contexts like monorepos, documentation trees, and datasets. The RLM tools allow the agent to load context, execute Python analysis code, and manage session state.

### RLM Tools

| Tool | Description |
|------|-------------|
| `rlm_load(path)` | Load context from file/directory (resets session) |
| `rlm_load_append(path)` | Add additional context (preserves state) |
| `rlm_exec(code)` | Execute Python code with access to context |
| `rlm_query(prompt)` | Quick BM25/routing search + sub-agent answer |
| `rlm_helpers_*` | Manage reusable Python helper functions |
| `rlm_memory_*` | Key-value memory for multi-pass workflows |

### Python Builtins in `rlm_exec`

| Builtin | Description |
|---------|-------------|
| `peek(start, end)` | Extract substring from context |
| `find(pattern, flags)` | Regex search returning `[(start, end), ...]` |
| `search(query, k=10)` | BM25 search returning scored chunks |
| `stats()` | Context statistics (chars, tokens, lines) |
| `session()` | Session manifest (sources, helpers, memory, budget) |
| `limits()` | Current execution limits |
| `budget()` | Remaining budget (tokens, sub-calls, time) |
| `llm_query(prompt, tools)` | Spawn sub-agent for LLM queries |
| `find_routes(topic)` | Search AGENTS.md routing hierarchy |

### Configuration

Configure RLM in `~/.codex/config.toml`:

```toml
[rlm.safety]
max_total_tokens = 1000000
max_sub_calls = 50
max_tool_calls_per_turn = 100
max_cpu_seconds = 300
allowed_modules = ["json", "re", "collections", "itertools", "functools"]

[rlm.budget]
max_cost_usd = 10.0
```

Model provider entries under `[model_providers]` are used for sub-agent LLM calls.

### Sub-Agent Tool Policy

By default, `llm_query` sub-agents can only use read-only tools: `read_file`, `list_dir`, `grep_files`.

To allow additional tools (e.g., shell access), configure `allowed_tool_overrides`:

```toml
[rlm.sub_agent_policy]
# Tools that Python code can request for sub-agents
allowed_tool_overrides = ["shell", "apply_patch", "web_search"]
```

When Python code calls `llm_query(prompt, tools=["shell"])`:
- If `shell` is in `allowed_tool_overrides`, the sub-agent gets shell access
- If not, a `PolicyViolationError` is raised
- `rlm_*` tools are always blocked to prevent recursive RLM calls

Tool override events are logged in the `rlm_exec` response under `tool_override_events`.

## JSON Schema

The generated JSON Schema for `config.toml` lives at `codex-rs/core/config.schema.json`.

## Notices

Codex stores "do not show again" flags for some UI prompts under the `[notice]` table.

Ctrl+C/Ctrl+D quitting uses a ~1 second double-press hint (`ctrl + c again to quit`).
