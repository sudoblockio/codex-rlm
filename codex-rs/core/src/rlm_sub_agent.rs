use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::anyhow;
use codex_rlm::BudgetSnapshot;
use codex_rlm::error::BudgetExceededKind;
use codex_rlm::estimate_tokens;
use codex_rlm::python::BatchCallResult;
use codex_rlm::python::LlmCallback;
use serde::Serialize;
use tokio::time::timeout;

use crate::agent::AgentStatus;
use crate::codex::Session;
use crate::codex::TurnContext;
use crate::config::Config;
use crate::features::Feature;
use crate::function_tool::FunctionCallError;
use codex_protocol::config_types::WebSearchMode;
use codex_protocol::openai_models::ConfigShellToolType;

/// Default tools for sub-agents: read_file, glob, grep (read-only, spec-defined names)
const DEFAULT_SUB_AGENT_TOOLS: [&str; 3] = ["read_file", "glob", "grep"];
const SUB_AGENT_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub(crate) struct ToolOverridePolicy {
    pub(crate) allowed_tool_overrides: Vec<String>,
    pub(crate) require_approval: bool,
}

pub(crate) fn default_sub_agent_tools() -> Vec<String> {
    DEFAULT_SUB_AGENT_TOOLS
        .iter()
        .map(|tool| (*tool).to_string())
        .collect()
}

pub(crate) fn tool_override_policy_json(config: &Config) -> Result<String> {
    let policy = match config
        .rlm
        .as_ref()
        .and_then(|rlm| rlm.sub_agent_policy.as_ref())
    {
        Some(policy) => ToolOverridePolicy {
            allowed_tool_overrides: policy.allowed_tool_overrides.clone().unwrap_or_default(),
            require_approval: policy.require_approval.unwrap_or(false),
        },
        None => ToolOverridePolicy {
            allowed_tool_overrides: Vec::new(),
            require_approval: false,
        },
    };

    Ok(serde_json::to_string(&policy)?)
}

pub(crate) fn build_sub_agent_config(
    turn: &TurnContext,
    tool_allowlist: &[String],
) -> Result<Config, FunctionCallError> {
    let base_config = turn.client.config();
    let mut config = (*base_config).clone();
    let tool_allowlist = expand_tool_allowlist(turn, tool_allowlist);

    config.model = Some(turn.client.get_model());
    config.model_provider = turn.client.get_provider();
    config.model_reasoning_effort = turn.client.get_reasoning_effort();
    config.model_reasoning_summary = turn.client.get_reasoning_summary();
    config.developer_instructions = turn.developer_instructions.clone();
    config.compact_prompt = turn.compact_prompt.clone();
    config.user_instructions = turn.user_instructions.clone();
    config.shell_environment_policy = turn.shell_environment_policy.clone();
    config.codex_linux_sandbox_exe = turn.codex_linux_sandbox_exe.clone();
    config.cwd = turn.cwd.clone();
    config
        .approval_policy
        .set(turn.approval_policy)
        .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
    config
        .sandbox_policy
        .set(turn.sandbox_policy.clone())
        .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;

    config.experimental_supported_tools = tool_allowlist.clone();
    config.tool_allowlist = Some(tool_allowlist.clone());

    config.features.disable(Feature::Collab);
    config.features.disable(Feature::CollaborationModes);

    let allow_shell = tool_allowlist.iter().any(|tool| {
        matches!(
            tool.as_str(),
            "shell" | "local_shell" | "shell_command" | "exec_command" | "write_stdin"
        )
    });
    if allow_shell {
        config.features.enable(Feature::ShellTool);
    } else {
        config.features.disable(Feature::ShellTool);
    }

    let allow_unified_exec = tool_allowlist
        .iter()
        .any(|tool| matches!(tool.as_str(), "exec_command" | "write_stdin"));
    if allow_unified_exec {
        config.features.enable(Feature::UnifiedExec);
    } else {
        config.features.disable(Feature::UnifiedExec);
    }

    if tool_allowlist
        .iter()
        .any(|tool| tool.as_str() == "apply_patch")
    {
        config.include_apply_patch_tool = true;
        config.features.enable(Feature::ApplyPatchFreeform);
    } else {
        config.include_apply_patch_tool = false;
        config.features.disable(Feature::ApplyPatchFreeform);
    }

    if tool_allowlist
        .iter()
        .any(|tool| tool.as_str() == "web_search")
    {
        config.web_search_mode = base_config.web_search_mode;
    } else {
        config.web_search_mode = Some(WebSearchMode::Disabled);
        config.features.disable(Feature::WebSearchRequest);
        config.features.disable(Feature::WebSearchCached);
    }

    Ok(config)
}

fn expand_tool_allowlist(turn: &TurnContext, tool_allowlist: &[String]) -> Vec<String> {
    let mut expanded = tool_allowlist.to_vec();
    let wants_shell = expanded.iter().any(|tool| {
        matches!(
            tool.as_str(),
            "shell" | "local_shell" | "shell_command" | "exec_command" | "write_stdin"
        )
    });

    if wants_shell {
        let shell_type = turn.client.get_model_info().shell_type;
        match shell_type {
            ConfigShellToolType::UnifiedExec => {
                if !expanded.iter().any(|tool| tool == "exec_command") {
                    expanded.push("exec_command".to_string());
                }
                if !expanded.iter().any(|tool| tool == "write_stdin") {
                    expanded.push("write_stdin".to_string());
                }
            }
            ConfigShellToolType::ShellCommand => {
                if !expanded.iter().any(|tool| tool == "shell_command") {
                    expanded.push("shell_command".to_string());
                }
            }
            ConfigShellToolType::Local => {
                if !expanded.iter().any(|tool| tool == "local_shell") {
                    expanded.push("local_shell".to_string());
                }
            }
            ConfigShellToolType::Default => {
                if !expanded.iter().any(|tool| tool == "shell") {
                    expanded.push("shell".to_string());
                }
            }
            ConfigShellToolType::Disabled => {}
        }
    }

    expanded
}

pub(crate) async fn run_sub_agent(
    session: &Session,
    turn: &TurnContext,
    prompt: String,
    tool_allowlist: Vec<String>,
) -> Result<String, FunctionCallError> {
    let config = build_sub_agent_config(turn, &tool_allowlist)?;
    let agent_id = session
        .services
        .agent_control
        .spawn_agent(config, prompt)
        .await
        .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;

    let mut status_rx = session
        .services
        .agent_control
        .subscribe_status(agent_id)
        .await
        .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;

    let result = timeout(SUB_AGENT_TIMEOUT, async {
        loop {
            let status = status_rx.borrow().clone();
            match status {
                AgentStatus::Completed(output) => {
                    return Ok(output.unwrap_or_else(|| "No output".to_string()));
                }
                AgentStatus::Errored(msg) => {
                    return Err(msg);
                }
                AgentStatus::Shutdown => {
                    return Err("Agent shutdown unexpectedly".to_string());
                }
                AgentStatus::NotFound => {
                    return Err("Agent not found".to_string());
                }
                AgentStatus::PendingInit | AgentStatus::Running => {
                    if status_rx.changed().await.is_err() {
                        return Err("Agent status channel closed".to_string());
                    }
                }
            }
        }
    })
    .await;

    let _ = session
        .services
        .agent_control
        .shutdown_agent(agent_id)
        .await;

    match result {
        Ok(Ok(output)) => Ok(output),
        Ok(Err(msg)) => Err(FunctionCallError::RespondToModel(msg)),
        Err(_) => Err(FunctionCallError::RespondToModel(
            "rlm_query timed out".to_string(),
        )),
    }
}

#[derive(Clone)]
pub(crate) struct RlmSubAgentCallback {
    session: Arc<Session>,
    turn: Arc<TurnContext>,
    budget_state: Arc<StdMutex<BudgetSnapshot>>,
    runtime_handle: tokio::runtime::Handle,
}

impl RlmSubAgentCallback {
    pub(crate) fn new(
        session: Arc<Session>,
        turn: Arc<TurnContext>,
        budget_state: Arc<StdMutex<BudgetSnapshot>>,
        runtime_handle: tokio::runtime::Handle,
    ) -> Self {
        Self {
            session,
            turn,
            budget_state,
            runtime_handle,
        }
    }

    fn reserve_budget(&self, token_cost: u64, sub_calls: u32) -> Result<()> {
        let mut guard = self
            .budget_state
            .lock()
            .map_err(|_| anyhow!("budget lock poisoned"))?;

        if guard.remaining_sub_calls < sub_calls {
            return Err(codex_rlm::RlmError::BudgetExceeded {
                kind: BudgetExceededKind::SubCalls,
                remaining: guard.remaining_sub_calls.into(),
                requested: sub_calls.into(),
            }
            .into());
        }

        if guard.remaining_tokens < token_cost {
            return Err(codex_rlm::RlmError::BudgetExceeded {
                kind: BudgetExceededKind::Tokens,
                remaining: guard.remaining_tokens,
                requested: token_cost,
            }
            .into());
        }

        guard.remaining_sub_calls -= sub_calls;
        guard.remaining_tokens -= token_cost;
        Ok(())
    }

    fn charge_time(&self, elapsed_ms: u64) -> Result<()> {
        let mut guard = self
            .budget_state
            .lock()
            .map_err(|_| anyhow!("budget lock poisoned"))?;

        if guard.remaining_ms < elapsed_ms {
            return Err(codex_rlm::RlmError::BudgetExceeded {
                kind: BudgetExceededKind::Time,
                remaining: guard.remaining_ms,
                requested: elapsed_ms,
            }
            .into());
        }

        guard.remaining_ms -= elapsed_ms;
        Ok(())
    }
}

impl LlmCallback for RlmSubAgentCallback {
    fn call(&self, prompt: &str, tools: Option<Vec<String>>) -> Result<String> {
        let tool_allowlist = tools.unwrap_or_else(default_sub_agent_tools);
        let token_cost = estimate_tokens(prompt);
        self.reserve_budget(token_cost, 1)?;

        let session = Arc::clone(&self.session);
        let turn = Arc::clone(&self.turn);
        let prompt = prompt.to_string();
        let start = Instant::now();
        let output = self
            .runtime_handle
            .block_on(async { run_sub_agent(&session, &turn, prompt, tool_allowlist).await })
            .map_err(|err| anyhow!(err.to_string()))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        self.charge_time(elapsed_ms)?;
        Ok(output)
    }

    fn batch(&self, prompts: Vec<String>, max_concurrent: usize) -> Result<Vec<BatchCallResult>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }

        let token_cost = prompts
            .iter()
            .map(|prompt| estimate_tokens(prompt))
            .sum::<u64>();
        let sub_calls = u32::try_from(prompts.len()).unwrap_or(u32::MAX);
        self.reserve_budget(token_cost, sub_calls)?;

        let session = Arc::clone(&self.session);
        let turn = Arc::clone(&self.turn);
        let tool_allowlist = default_sub_agent_tools();
        let start = Instant::now();

        // Run sub-agents with concurrency limit
        let results = self.runtime_handle.block_on(async {
            use tokio::sync::Semaphore;

            let semaphore = Arc::new(Semaphore::new(max_concurrent.max(1)));

            let futures: Vec<_> = prompts
                .into_iter()
                .map(|prompt| {
                    let session = Arc::clone(&session);
                    let turn = Arc::clone(&turn);
                    let tool_allowlist = tool_allowlist.clone();
                    let semaphore = Arc::clone(&semaphore);
                    async move {
                        let _permit = semaphore.acquire().await;
                        run_sub_agent(&session, &turn, prompt, tool_allowlist).await
                    }
                })
                .collect();

            // Execute all futures concurrently (respecting semaphore limit)
            futures::future::join_all(futures).await
        });

        // Convert each result to BatchCallResult
        let outputs: Vec<BatchCallResult> = results
            .into_iter()
            .map(|result| match result {
                Ok(response) => BatchCallResult::success(response),
                Err(err) => {
                    // Categorize the error
                    let err_str = err.to_string();
                    let (code, retriable) =
                        if err_str.contains("timeout") || err_str.contains("timed out") {
                            ("timeout", true)
                        } else if err_str.contains("budget") {
                            ("budget_exceeded", false)
                        } else {
                            ("sub_agent_error", true)
                        };
                    BatchCallResult::error(code, err_str, retriable)
                }
            })
            .collect();

        let elapsed_ms = start.elapsed().as_millis() as u64;
        self.charge_time(elapsed_ms)?;
        Ok(outputs)
    }

    fn budget_snapshot(&self) -> Result<BudgetSnapshot> {
        let guard = self
            .budget_state
            .lock()
            .map_err(|_| anyhow!("budget lock poisoned"))?;
        Ok(guard.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to test budget reservation logic directly on the budget state.
    fn reserve_budget(
        budget_state: &Arc<StdMutex<BudgetSnapshot>>,
        token_cost: u64,
        sub_calls: u32,
    ) -> Result<()> {
        let mut guard = budget_state.lock().map_err(|_| anyhow!("lock poisoned"))?;

        if guard.remaining_sub_calls < sub_calls {
            return Err(codex_rlm::RlmError::BudgetExceeded {
                kind: BudgetExceededKind::SubCalls,
                remaining: guard.remaining_sub_calls.into(),
                requested: sub_calls.into(),
            }
            .into());
        }

        if guard.remaining_tokens < token_cost {
            return Err(codex_rlm::RlmError::BudgetExceeded {
                kind: BudgetExceededKind::Tokens,
                remaining: guard.remaining_tokens,
                requested: token_cost,
            }
            .into());
        }

        guard.remaining_sub_calls -= sub_calls;
        guard.remaining_tokens -= token_cost;
        Ok(())
    }

    fn charge_time(budget_state: &Arc<StdMutex<BudgetSnapshot>>, elapsed_ms: u64) -> Result<()> {
        let mut guard = budget_state.lock().map_err(|_| anyhow!("lock poisoned"))?;

        if guard.remaining_ms < elapsed_ms {
            return Err(codex_rlm::RlmError::BudgetExceeded {
                kind: BudgetExceededKind::Time,
                remaining: guard.remaining_ms,
                requested: elapsed_ms,
            }
            .into());
        }

        guard.remaining_ms -= elapsed_ms;
        Ok(())
    }

    #[test]
    fn reserve_budget_decrements_counters() {
        let budget = BudgetSnapshot::new(1000, 10, 100, 60000);
        let budget_state = Arc::new(StdMutex::new(budget));

        reserve_budget(&budget_state, 100, 2).unwrap();

        let snapshot = budget_state.lock().unwrap().clone();
        assert_eq!(snapshot.remaining_tokens, 900);
        assert_eq!(snapshot.remaining_sub_calls, 8);
    }

    #[test]
    fn reserve_budget_fails_when_sub_calls_exhausted() {
        let budget = BudgetSnapshot::new(1000, 1, 100, 60000);
        let budget_state = Arc::new(StdMutex::new(budget));

        // First call should succeed
        reserve_budget(&budget_state, 100, 1).unwrap();

        // Second call should fail - no sub_calls remaining
        let err = reserve_budget(&budget_state, 100, 1).unwrap_err();
        assert!(err.to_string().contains("budget exceeded"));
    }

    #[test]
    fn reserve_budget_fails_when_tokens_exhausted() {
        let budget = BudgetSnapshot::new(100, 10, 100, 60000);
        let budget_state = Arc::new(StdMutex::new(budget));

        // Request more tokens than available
        let err = reserve_budget(&budget_state, 200, 1).unwrap_err();
        assert!(err.to_string().contains("budget exceeded"));
    }

    #[test]
    fn charge_time_decrements_remaining_ms() {
        let budget = BudgetSnapshot::new(1000, 10, 100, 5000);
        let budget_state = Arc::new(StdMutex::new(budget));

        charge_time(&budget_state, 2000).unwrap();

        let snapshot = budget_state.lock().unwrap().clone();
        assert_eq!(snapshot.remaining_ms, 3000);
    }

    #[test]
    fn charge_time_fails_when_time_exhausted() {
        let budget = BudgetSnapshot::new(1000, 10, 100, 1000);
        let budget_state = Arc::new(StdMutex::new(budget));

        // Request more time than available
        let err = charge_time(&budget_state, 2000).unwrap_err();
        assert!(err.to_string().contains("budget exceeded"));
    }

    #[test]
    fn batch_budget_reserves_all_prompts_at_once() {
        // Simulate batch of 3 prompts with ~100 tokens each
        let budget = BudgetSnapshot::new(500, 5, 100, 60000);
        let budget_state = Arc::new(StdMutex::new(budget));

        // Reserve for 3 prompts (300 tokens, 3 sub_calls)
        reserve_budget(&budget_state, 300, 3).unwrap();

        let snapshot = budget_state.lock().unwrap().clone();
        assert_eq!(snapshot.remaining_tokens, 200);
        assert_eq!(snapshot.remaining_sub_calls, 2);
    }

    #[test]
    fn batch_budget_fails_if_insufficient_sub_calls() {
        // Only 2 sub_calls available but trying to run 3
        let budget = BudgetSnapshot::new(1000, 2, 100, 60000);
        let budget_state = Arc::new(StdMutex::new(budget));

        let err = reserve_budget(&budget_state, 300, 3).unwrap_err();
        assert!(err.to_string().contains("budget exceeded"));

        // Budget should be unchanged
        let snapshot = budget_state.lock().unwrap().clone();
        assert_eq!(snapshot.remaining_sub_calls, 2);
    }

    #[test]
    fn tool_override_policy_serializes_correctly() {
        let policy = ToolOverridePolicy {
            allowed_tool_overrides: vec!["shell".to_string(), "apply_patch".to_string()],
            require_approval: false,
        };
        let json = serde_json::to_string(&policy).unwrap();
        let parsed: ToolOverridePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.allowed_tool_overrides.len(), 2);
        assert!(parsed.allowed_tool_overrides.contains(&"shell".to_string()));
        assert!(!parsed.require_approval);
    }

    #[test]
    fn tool_override_policy_empty_by_default() {
        // When no policy is set, we get empty allowed_tool_overrides
        let policy = ToolOverridePolicy {
            allowed_tool_overrides: Vec::new(),
            require_approval: false,
        };
        assert!(policy.allowed_tool_overrides.is_empty());
    }

    #[test]
    fn default_sub_agent_tools_are_read_only() {
        let tools = default_sub_agent_tools();
        // Default read-only tools: read_file, glob, grep (spec-defined names)
        assert!(tools.contains(&"read_file".to_string()));
        assert!(tools.contains(&"glob".to_string()));
        assert!(tools.contains(&"grep".to_string()));
        // Should NOT contain write tools
        assert!(!tools.contains(&"shell".to_string()));
        assert!(!tools.contains(&"apply_patch".to_string()));
    }

    #[test]
    fn concurrent_budget_reservations_are_atomic() {
        // Test that concurrent budget reservations don't corrupt state
        use std::thread;

        let budget = BudgetSnapshot::new(10000, 100, 100, 60000);
        let budget_state = Arc::new(StdMutex::new(budget));

        // Spawn 10 threads, each making 10 reservations of 1 sub_call
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let state = Arc::clone(&budget_state);
                thread::spawn(move || {
                    for _ in 0..10 {
                        let _ = reserve_budget(&state, 10, 1);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have used exactly 100 sub_calls (10 threads × 10 calls)
        let snapshot = budget_state.lock().unwrap().clone();
        assert_eq!(snapshot.remaining_sub_calls, 0);
        assert_eq!(snapshot.remaining_tokens, 9000); // 10000 - (100 × 10)
    }

    #[test]
    fn concurrent_budget_exhaustion_is_safe() {
        // Test that concurrent reservations don't go negative when budget is limited
        use std::sync::atomic::AtomicU32;
        use std::sync::atomic::Ordering;
        use std::thread;

        let budget = BudgetSnapshot::new(10000, 5, 100, 60000);
        let budget_state = Arc::new(StdMutex::new(budget));
        let success_count = Arc::new(AtomicU32::new(0));

        // Spawn 10 threads, each trying to reserve 1 sub_call
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let state = Arc::clone(&budget_state);
                let counter = Arc::clone(&success_count);
                thread::spawn(move || {
                    if reserve_budget(&state, 10, 1).is_ok() {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Exactly 5 should succeed (the budget limit)
        assert_eq!(success_count.load(Ordering::SeqCst), 5);

        // Budget should not go negative
        let snapshot = budget_state.lock().unwrap().clone();
        assert_eq!(snapshot.remaining_sub_calls, 0);
    }

    #[test]
    fn concurrent_time_charging_is_atomic() {
        use std::thread;

        let budget = BudgetSnapshot::new(10000, 100, 100, 10000);
        let budget_state = Arc::new(StdMutex::new(budget));

        // Spawn 10 threads, each charging 100ms
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let state = Arc::clone(&budget_state);
                thread::spawn(move || {
                    charge_time(&state, 100).unwrap();
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have charged exactly 1000ms
        let snapshot = budget_state.lock().unwrap().clone();
        assert_eq!(snapshot.remaining_ms, 9000);
    }

    #[tokio::test]
    async fn parallel_batch_reserves_budget_atomically() {
        // Verify that batch budget reservation happens before parallel execution starts
        let budget = BudgetSnapshot::new(1000, 5, 100, 60000);
        let budget_state = Arc::new(StdMutex::new(budget));

        // Reserve for a batch of 3 prompts
        reserve_budget(&budget_state, 300, 3).unwrap();

        // After reservation, budget should reflect all 3 sub_calls
        let snapshot = budget_state.lock().unwrap().clone();
        assert_eq!(snapshot.remaining_sub_calls, 2);
        assert_eq!(snapshot.remaining_tokens, 700);
    }
}
