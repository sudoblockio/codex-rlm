use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;

use crate::function_tool::FunctionCallError;
use crate::rlm_session::RlmExecOutcome;
use crate::rlm_session::RlmLimitsOverride;
use crate::rlm_sub_agent::RlmSubAgentCallback;
use crate::rlm_sub_agent::tool_override_policy_json;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::rlm_types::error_value;
use crate::tools::handlers::rlm_types::json_tool_output;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub(crate) struct RlmExecHandler;

#[derive(Deserialize)]
struct RlmExecArgs {
    code: String,
    #[serde(default)]
    limits_override: Option<RlmLimitsOverride>,
}

#[async_trait]
impl ToolHandler for RlmExecHandler {
    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<crate::tools::context::ToolOutput, FunctionCallError> {
        let ToolInvocation {
            payload,
            session,
            turn,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "rlm_exec handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: RlmExecArgs = parse_arguments(&arguments)?;
        let code = args.code.trim();
        if code.is_empty() {
            let value = error_value(
                "python_error",
                "code must not be empty",
                Some("Provide Python code to execute"),
            );
            return Ok(json_tool_output(value, false));
        }

        let rlm_session = session
            .rlm_session()
            .await
            .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
        let has_context = {
            let guard = rlm_session.lock().await;
            guard.has_context()
        };
        if !has_context {
            let value = error_value(
                "context_not_loaded",
                "rlm_load must be called before rlm_exec",
                Some("Call rlm_load with a path before executing code"),
            );
            return Ok(json_tool_output(value, false));
        }

        let code = code.to_string();
        let limits_override = args.limits_override.clone();
        let budget_state = {
            let guard = rlm_session.lock().await;
            guard.budget_state()
        };
        let callback = Arc::new(RlmSubAgentCallback::new(
            Arc::clone(&session),
            Arc::clone(&turn),
            budget_state,
            tokio::runtime::Handle::current(),
        ));
        let policy_json = tool_override_policy_json(turn.client.config().as_ref())
            .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
        let start = Instant::now();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut guard = rlm_session.blocking_lock();
            guard.exec(
                &code,
                limits_override.as_ref(),
                Some(callback),
                Some(&policy_json),
            )
        })
        .await
        .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?
        .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
        let execution_time_ms = start.elapsed().as_millis() as u64;

        Ok(build_exec_response(outcome, execution_time_ms))
    }
}

fn build_exec_response(
    outcome: RlmExecOutcome,
    execution_time_ms: u64,
) -> crate::tools::context::ToolOutput {
    let mut warnings = Vec::new();
    if outcome.result.truncated {
        warnings.push("output_truncated");
    }
    if outcome.result.find_results_capped {
        warnings.push("find_results_capped");
    }
    if outcome.result.result_json_error.is_some() {
        warnings.push("result_not_serializable");
    }
    if outcome.result.tool_override_denied {
        warnings.push("tool_override_denied");
    }

    let result_json = parse_json_value(outcome.result.result_json.as_deref());
    let result_meta = parse_json_value(outcome.result.result_meta_json.as_deref());

    let tool_override_events: Vec<Value> =
        parse_json_array(outcome.result.tool_override_events_json.as_deref());

    if outcome.result.timed_out {
        let value = json!({
            "success": false,
            "error_code": "python_timeout",
            "error_message": outcome
                .result
                .error
                .clone()
                .unwrap_or_else(|| "Python execution timed out".to_string()),
            "suggestion": "Optimize the code or request higher limits",
            "stdout": outcome.result.output,
            "stderr": "",
            "warnings": warnings,
        });
        return json_tool_output(value, false);
    }

    if let Some(error) = outcome.result.error {
        // Check if this is a policy violation error
        let (error_code, suggestion) = if outcome.result.tool_override_denied
            || error.contains("tool override denied")
            || error.contains("PolicyViolationError")
        {
            (
                "policy_violation",
                "Remove tools parameter or request approval from policy",
            )
        } else {
            (
                "python_error",
                "Check the traceback and fix the Python code",
            )
        };
        let value = json!({
            "success": false,
            "error_code": error_code,
            "error_message": error,
            "traceback": outcome.result.traceback,
            "suggestion": suggestion,
            "stdout": outcome.result.output,
            "stderr": "",
            "warnings": warnings,
        });
        return json_tool_output(value, false);
    }

    let value = json!({
        "success": true,
        "stdout": outcome.result.output,
        "stderr": "",
        "result_json": result_json,
        "result_meta": result_meta,
        "truncated": outcome.result.truncated,
        "warnings": warnings,
        "execution_time_ms": execution_time_ms,
        "limits_applied": outcome.limits_applied,
        "tool_override_events": tool_override_events,
        "budget": outcome.budget,
    });
    json_tool_output(value, true)
}

fn parse_json_value(input: Option<&str>) -> Value {
    match input {
        Some(raw) => serde_json::from_str(raw).unwrap_or(Value::Null),
        None => Value::Null,
    }
}

fn parse_json_array(input: Option<&str>) -> Vec<Value> {
    match input {
        Some(raw) => serde_json::from_str(raw).unwrap_or_default(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlm_session::RlmLimits;
    use crate::tools::context::ToolOutput;
    use codex_rlm::BudgetSnapshot;
    use codex_rlm::python::ExecutionResult;
    use pretty_assertions::assert_eq;

    fn base_result() -> ExecutionResult {
        ExecutionResult {
            output: "ok".to_string(),
            error: None,
            traceback: None,
            truncated: false,
            timed_out: false,
            find_results_capped: false,
            result_json: None,
            result_meta_json: None,
            result_json_error: None,
            tool_override_events_json: None,
            tool_override_denied: false,
            files_accessed: Vec::new(),
        }
    }

    #[test]
    fn exec_response_success_includes_result_json() {
        let mut result = base_result();
        result.result_json = Some(r#"{"k":1}"#.to_string());
        let outcome = RlmExecOutcome {
            result,
            limits_applied: RlmLimits::default(),
            budget: BudgetSnapshot::new(1, 1, 1, 1),
        };
        let output = build_exec_response(outcome, 12);
        let ToolOutput::Function { content, .. } = output else {
            panic!("expected function output");
        };
        let value: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["result_json"], json!({"k": 1}));
    }

    #[test]
    fn exec_response_warns_on_truncation_and_find_cap() {
        let mut result = base_result();
        result.truncated = true;
        result.find_results_capped = true;
        result.result_json_error = Some("result: nope".to_string());
        result.tool_override_denied = true;
        let outcome = RlmExecOutcome {
            result,
            limits_applied: RlmLimits::default(),
            budget: BudgetSnapshot::new(1, 1, 1, 1),
        };
        let output = build_exec_response(outcome, 5);
        let ToolOutput::Function { content, .. } = output else {
            panic!("expected function output");
        };
        let value: Value = serde_json::from_str(&content).unwrap();
        let warnings = value["warnings"].as_array().unwrap();
        assert_eq!(warnings.len(), 4);
        assert!(warnings.contains(&json!("output_truncated")));
        assert!(warnings.contains(&json!("find_results_capped")));
        assert!(warnings.contains(&json!("result_not_serializable")));
        assert!(warnings.contains(&json!("tool_override_denied")));
    }

    #[test]
    fn exec_response_timeout_returns_error_code() {
        let mut result = base_result();
        result.timed_out = true;
        result.error = Some("execution timed out".to_string());
        let outcome = RlmExecOutcome {
            result,
            limits_applied: RlmLimits::default(),
            budget: BudgetSnapshot::new(1, 1, 1, 1),
        };
        let output = build_exec_response(outcome, 40);
        let ToolOutput::Function { content, .. } = output else {
            panic!("expected function output");
        };
        let value: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(value["error_code"], "python_timeout");
    }

    #[test]
    fn exec_response_python_error_returns_error_code() {
        let mut result = base_result();
        result.error = Some("NameError".to_string());
        let outcome = RlmExecOutcome {
            result,
            limits_applied: RlmLimits::default(),
            budget: BudgetSnapshot::new(1, 1, 1, 1),
        };
        let output = build_exec_response(outcome, 7);
        let ToolOutput::Function { content, .. } = output else {
            panic!("expected function output");
        };
        let value: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(value["error_code"], "python_error");
    }

    #[test]
    fn exec_response_policy_violation_returns_correct_error_code() {
        let mut result = base_result();
        result.error = Some("tool override denied".to_string());
        result.tool_override_denied = true;
        let outcome = RlmExecOutcome {
            result,
            limits_applied: RlmLimits::default(),
            budget: BudgetSnapshot::new(1, 1, 1, 1),
        };
        let output = build_exec_response(outcome, 7);
        let ToolOutput::Function { content, .. } = output else {
            panic!("expected function output");
        };
        let value: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(value["error_code"], "policy_violation");
        assert!(value["suggestion"]
            .as_str()
            .unwrap()
            .contains("request approval"));
    }

    #[test]
    fn exec_response_includes_tool_override_events() {
        let mut result = base_result();
        result.tool_override_events_json =
            Some(r#"[{"requested_tools":["shell"],"granted":false}]"#.to_string());
        let outcome = RlmExecOutcome {
            result,
            limits_applied: RlmLimits::default(),
            budget: BudgetSnapshot::new(1, 1, 1, 1),
        };
        let output = build_exec_response(outcome, 3);
        let ToolOutput::Function { content, .. } = output else {
            panic!("expected function output");
        };
        let value: Value = serde_json::from_str(&content).unwrap();
        assert_eq!(
            value["tool_override_events"],
            json!([{"requested_tools":["shell"],"granted":false}])
        );
    }
}
