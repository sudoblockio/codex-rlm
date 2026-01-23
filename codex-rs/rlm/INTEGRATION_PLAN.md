# RLM Integration Plan v7

> **Status: COMPLETED** (January 2026)
>
> This plan has been implemented. RLM is now embedded as Codex tools rather than
> a standalone binary. See `docs/config.md` for usage documentation.
>
> ## Design Decisions
>
> **Helper execution in rlm_query: NOT IMPLEMENTED**
> - Decision: Keep rlm_query as a simple convenience tool (search → sub-agent → answer)
> - Rationale: Users who need pre-processing can use `rlm_exec` with `search()` + helpers
> - This maintains clear separation: rlm_query (simple), rlm_exec (programmable)

---

## Goal

Give the Codex agent the ability to process arbitrarily large contexts (monorepos, doc trees, datasets) by exposing Python-based exploration tools with structured output, session state, and multi-pass workflow support.

**Full spec:** `/Users/justin/codex/docs/rlm-runtime-spec.md`

---

## Quick Summary

```
rlm_load(path)           → Load context (resets session)
rlm_load_append(path)    → Add context (preserves state)
rlm_exec(code)           → Execute Python, return result_json
rlm_query(prompt)        → Quick scan+summarize (no Python)
rlm_helpers_*            → Session helper management
rlm_memory_*             → Session memory for multi-pass
```

**Key insight:** The agent writes Python code. No hidden LLM loop inside RLM.

---

## Sprint 1 TODO (Foundation)

- [x] Replace `rlm_analyze` with `rlm_load` + `rlm_exec` tool specs + handlers
- [x] Implement structured output for `rlm_exec` (`result_json`, `result_meta`, `warnings`, `tool_override_events`)
- [x] Add per-call `limits_override` and return `limits_applied`
- [x] Implement `rlm_load_append` (preserve helpers/memory/budget, update sources)
- [x] Add `session()` builtin returning manifest (sources, helpers, memory keys, budget, limits)
- [x] Add `limits()` builtin (expose current caps)
- [x] Add `rlm_memory_batch` tool (batch get/put)
- [x] Wire basic budget snapshot into `rlm_exec` responses
- [x] Tests: path validation, structured output, append behavior, session manifest, memory batch

---

## Phase 1a: Core Tools (MVP)

**Goal:** Ship something usable quickly.

### Deliverables
- [x] `rlm_load` handler with structured JSON response
- [x] `rlm_exec` handler with structured JSON response
- [x] `result_json` return channel (via `result` variable)
- [ ] Path validation against sandbox roots
- [x] Basic error taxonomy (`context_not_loaded`, `path_outside_sandbox`, `python_error`)
- [x] Output truncation with `truncated` flag and `warnings` array

### Files to Create/Modify
```
core/Cargo.toml                         # Add codex-rlm dependency
core/src/tools/handlers/rlm_types.rs    # RlmLoadResult, RlmExecResult, RlmError
core/src/tools/handlers/rlm_load.rs     # Handler
core/src/tools/handlers/rlm_exec.rs     # Handler with spawn_blocking
core/src/tools/handlers/rlm_session.rs  # RlmSession struct
core/src/tools/spec.rs                  # Tool specs
core/src/tools/mod.rs                   # Register handlers
```

### Acceptance Criteria
- [x] `rlm_load("/valid/path")` returns `{ success: true, stats: {...} }`
- [ ] `rlm_load("/outside/sandbox")` returns `{ success: false, error_code: "path_outside_sandbox" }`
- [x] `rlm_exec("result = {'k': 1}")` returns `{ success: true, result_json: {"k":1} }`
- [x] `rlm_exec("1/0")` returns `{ success: false, error_code: "python_error", traceback: "..." }`
- [x] Output > 100KB returns `{ truncated: true, warnings: ["output_truncated"] }`
- [x] `rlm_exec` without prior `rlm_load` returns `{ error_code: "context_not_loaded" }`

---

## Phase 1b: Agent Ergonomics

**Goal:** Make it a joy to use.

### Deliverables
- [x] `rlm_load_append` handler
- [x] `rlm_query` convenience tool
- [x] `limits()` builtin
- [ ] Full error taxonomy with `suggestion` field
- [x] `warnings` array for non-fatal signals

### Files to Create/Modify
```
core/src/tools/handlers/rlm_load_append.rs
core/src/tools/handlers/rlm_query.rs
rlm/src/python.rs                       # Add limits() builtin
```

### Acceptance Criteria
- [x] `rlm_load_append` extends context without clearing state
- [x] `stats().sources` array grows with each append
- [x] `rlm_query("find TODOs")` returns structured answer
- [x] `limits()` returns `{"max_output_bytes", "max_find_results", ...}`
- [ ] All errors include `suggestion` field

---

## Phase 2: Sub-Agent Integration

**Goal:** Make `llm_query` spawn real Codex sub-agents.

### Deliverables
- [x] `LlmCallback` trait in codex-rlm with core callback impl
- [x] Callback wired to `AgentControl::spawn_agent()`
- [x] GIL release during sub-agent wait (via `py.detach`)
- [x] Budget tracking (sub_calls, tokens)
- [x] Sub-agent tool scoping (read-only by default)
- [x] `tools` parameter with policy-gated override

### Files to Create/Modify
```
core/src/tools/handlers/rlm_callback.rs  # LlmCallback implementation
core/src/tools/handlers/rlm_budget.rs    # BudgetTracker
rlm/src/python.rs                        # Update llm_query to use callback
```

### GIL Strategy
```rust
fn llm_query(py: Python<'_>, prompt: String, tools: Option<Vec<String>>) -> PyResult<String> {
    py.allow_threads(|| {
        tokio::task::block_in_place(|| {
            Handle::current().block_on(async {
                callback.spawn_and_wait(prompt, tools).await
            })
        })
    })
}
```

### Acceptance Criteria
- [x] `llm_query("2+2?")` returns sub-agent response
- [ ] `llm_query_batch([...])` runs in parallel
- [x] Sub-agent cannot invoke shell (unless override)
- [x] `llm_query(..., tools=["shell"])` checked against policy
- [x] `budget()` reflects remaining sub_calls
- [ ] No deadlocks under concurrent tool calls

---

## Phase 3: Helpers & Memory

**Goal:** Enable multi-pass workflows.

### Deliverables
- [x] `rlm_helpers_add`, `rlm_helpers_list`, `rlm_helpers_remove`
- [x] `rlm_memory_put`, `rlm_memory_get`, `rlm_memory_list`, `rlm_memory_clear`
- [x] Storage limits (1MB helpers, 5MB memory)
- [x] Proper lifecycle (cleared on `rlm_load`, preserved on `rlm_load_append`)

### Files to Create/Modify
```
core/src/tools/handlers/rlm_helpers.rs
core/src/tools/handlers/rlm_memory.rs
core/src/tools/handlers/rlm_session.rs   # Add helper/memory storage
```

### Acceptance Criteria
- [x] Helper defined in one `rlm_exec` is available in next
- [x] Memory persists across `rlm_exec` calls
- [x] `rlm_load` clears helpers and memory
- [x] `rlm_load_append` preserves helpers and memory
- [x] Exceeding storage limits returns appropriate error

---

## Phase 4: Polish & Cleanup

**Goal:** Production-ready quality.

### Deliverables
- [x] Remove old `rlm_analyze` tool handler
- [x] Remove subprocess-based code
- [ ] Cancellation support
- [ ] Comprehensive test coverage
- [ ] Documentation update
- [x] Config schema update

### Files to Remove/Modify
```
core/src/tools/handlers/rlm.rs           # Remove old handler
```

### Acceptance Criteria
- [x] No subprocess spawning in production path
- [ ] Cancellation flag checked, returns clean error
- [ ] All new tools documented
- [x] Config schema includes `[rlm]` section

---

## Testing Strategy

### Unit Tests
```
[ ] Path validation (sandbox, symlinks, traversal)
[ ] Python builtins (peek, find, stats, limits, budget)
[x] result_json extraction
[x] Output truncation
[x] Error code mapping
[ ] Helper storage and limits
[ ] Memory storage and limits
```

### Integration Tests
```
[ ] Full rlm_load → rlm_exec round-trip
[ ] rlm_load_append preserves state
[ ] llm_query spawns sub-agent
[ ] llm_query_batch parallel execution
[ ] Budget enforcement
[ ] Session isolation (concurrent agents)
[ ] Cancellation handling
```

### Real Workload Tests
```
[ ] Load Linux kernel docs (1M+ tokens)
[ ] Find all mutex_lock usage, analyze top 10
[ ] Multi-pass workflow: scan → analyze → synthesize
[ ] Performance: load < 3s, exec < 100ms (excluding LLM)
```

---

## Open Questions (Resolved)

| Question | Decision |
|----------|----------|
| Explicit `rlm_reset` tool? | No. `rlm_load` resets. |
| Structured vs string output? | Structured JSON with `result_json`. |
| Sub-agent tool access? | Read-only default, policy-gated override. |
| Budget reset on `rlm_load`? | Yes. |
| Context layering? | `rlm_load_append`. |
| Convenience tool? | `rlm_query` for simple workflows. |
| Visibility into limits? | `limits()` builtin. |
| Session state? | Helpers + memory, cleared on `rlm_load`. |

---

## API Adjustments Needed in `codex-rlm` Crate

- [x] Export `PythonRuntime` and `ContextStore` publicly
- [x] Add `set_llm_callback(Box<dyn Fn(String, Option<Vec<String>>) -> Result<String>>)`
- [x] Return `RlmExecResult` struct (not just stdout string)
- [x] Add `limits()` builtin
- [x] Add budget tracking hooks
- [ ] Support helper injection before exec
- [ ] Support context append without reset

---

## Dependency Graph

```
Phase 1a (Core)
    │
    ├── Phase 1b (Ergonomics)
    │       │
    │       └── Phase 3 (Helpers/Memory)
    │
    └── Phase 2 (Sub-Agents)
            │
            └── Phase 4 (Polish)
```

Phases 1b and 2 can run in parallel after 1a.
Phase 3 depends on session state from 1b.
Phase 4 is final cleanup after everything works.

---

## Success Metrics

**Phase 1a complete:** Agent can load context and execute Python with structured output.

**Phase 1b complete:** Agent has smooth ergonomics (append, query, limits).

**Phase 2 complete:** `llm_query` spawns real sub-agents with budget tracking.

**Phase 3 complete:** Multi-pass workflows work with helpers and memory.

**Phase 4 complete:** Production-ready, no dead code, fully tested.
