use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;

use crate::function_tool::FunctionCallError;
use crate::rlm_session::RlmSession;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::rlm_types::error_value;
use crate::tools::handlers::rlm_types::json_tool_output;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub(crate) struct RlmMemoryHandler;

#[derive(Deserialize)]
struct RlmMemoryPutArgs {
    key: String,
    value: Value,
}

#[derive(Deserialize)]
struct RlmMemoryGetArgs {
    key: String,
}

#[derive(Deserialize)]
struct RlmMemoryBatchArgs {
    ops: Vec<RlmMemoryBatchOp>,
}

#[derive(Deserialize)]
struct RlmMemoryBatchOp {
    op: String,
    key: String,
    #[serde(default)]
    value: Option<Value>,
}

#[async_trait]
impl ToolHandler for RlmMemoryHandler {
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
            tool_name,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "rlm_memory handler received unsupported payload".to_string(),
                ));
            }
        };

        let rlm_session = session
            .rlm_session()
            .await
            .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
        let mut guard = rlm_session.lock().await;

        match tool_name.as_str() {
            "rlm_memory_put" => {
                let args: RlmMemoryPutArgs = parse_arguments(&arguments)?;
                match guard.memory_put(args.key.clone(), args.value) {
                    Ok(()) => Ok(json_tool_output(
                        json!({"success": true, "key": args.key}),
                        true,
                    )),
                    Err(err) => {
                        let value = error_value(
                            "memory_limit_exceeded",
                            err.to_string(),
                            Some("Clear memory or reduce stored values"),
                        );
                        Ok(json_tool_output(value, false))
                    }
                }
            }
            "rlm_memory_get" => {
                let args: RlmMemoryGetArgs = parse_arguments(&arguments)?;
                let value = guard.memory_get(&args.key);
                Ok(json_tool_output(
                    json!({"success": true, "key": args.key, "value": value}),
                    true,
                ))
            }
            "rlm_memory_list" => {
                let keys = guard.memory_keys();
                Ok(json_tool_output(
                    json!({"success": true, "keys": keys}),
                    true,
                ))
            }
            "rlm_memory_clear" => match guard.memory_clear() {
                Ok(()) => Ok(json_tool_output(json!({"success": true}), true)),
                Err(err) => {
                    let value = error_value(
                        "memory_error",
                        err.to_string(),
                        Some("Retry clearing memory"),
                    );
                    Ok(json_tool_output(value, false))
                }
            },
            "rlm_memory_batch" => {
                let args: RlmMemoryBatchArgs = parse_arguments(&arguments)?;
                let results = apply_memory_batch(&mut guard, args.ops);
                Ok(json_tool_output(
                    json!({"success": true, "results": results}),
                    true,
                ))
            }
            _ => {
                let value = error_value(
                    "unsupported_tool",
                    format!("unsupported rlm memory tool: {tool_name}"),
                    Some("Use rlm_memory_put, rlm_memory_get, rlm_memory_list, rlm_memory_clear, or rlm_memory_batch"),
                );
                Ok(json_tool_output(value, false))
            }
        }
    }
}

fn apply_memory_batch(session: &mut RlmSession, ops: Vec<RlmMemoryBatchOp>) -> Vec<Value> {
    let mut results = Vec::with_capacity(ops.len());
    for op in ops {
        match op.op.as_str() {
            "put" => {
                let Some(value) = op.value else {
                    results.push(json!({
                        "op": "put",
                        "key": op.key,
                        "ok": false,
                        "error": "missing value"
                    }));
                    continue;
                };
                let result = session.memory_put(op.key.clone(), value);
                match result {
                    Ok(()) => results.push(json!({"op": "put", "key": op.key, "ok": true})),
                    Err(err) => results.push(json!({
                        "op": "put",
                        "key": op.key,
                        "ok": false,
                        "error": err.to_string()
                    })),
                }
            }
            "get" => {
                let value = session.memory_get(&op.key);
                results.push(json!({
                    "op": "get",
                    "key": op.key,
                    "value": value
                }));
            }
            other => {
                results.push(json!({
                    "op": other,
                    "key": op.key,
                    "ok": false,
                    "error": "unknown op"
                }));
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn memory_batch_put_get_round_trips() {
        let mut session = RlmSession::new().unwrap();
        let ops = vec![
            RlmMemoryBatchOp {
                op: "put".to_string(),
                key: "alpha".to_string(),
                value: Some(json!({"x": 1})),
            },
            RlmMemoryBatchOp {
                op: "get".to_string(),
                key: "alpha".to_string(),
                value: None,
            },
            RlmMemoryBatchOp {
                op: "get".to_string(),
                key: "missing".to_string(),
                value: None,
            },
        ];

        let results = apply_memory_batch(&mut session, ops);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0]["ok"], true);
        assert_eq!(results[1]["value"], json!({"x": 1}));
        assert_eq!(results[2]["value"], Value::Null);
    }
}
