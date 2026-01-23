use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::function_tool::FunctionCallError;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::rlm_types::error_value;
use crate::tools::handlers::rlm_types::json_tool_output;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub(crate) struct RlmHelpersHandler;

#[derive(Deserialize)]
struct RlmHelpersAddArgs {
    name: String,
    code: String,
}

#[derive(Deserialize)]
struct RlmHelpersRemoveArgs {
    name: String,
}

#[derive(Deserialize)]
struct EmptyArgs {}

#[async_trait]
impl ToolHandler for RlmHelpersHandler {
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
                    "rlm_helpers handler received unsupported payload".to_string(),
                ));
            }
        };

        match tool_name.as_str() {
            "rlm_helpers_add" => {
                let args: RlmHelpersAddArgs = parse_arguments(&arguments)?;
                let name = args.name.trim().to_string();
                let code = args.code.trim().to_string();
                if name.is_empty() || code.is_empty() {
                    let value = error_value(
                        "invalid_argument",
                        "name and code must not be empty",
                        Some("Provide a helper name and non-empty code"),
                    );
                    return Ok(json_tool_output(value, false));
                }

                let rlm_session = session
                    .rlm_session()
                    .await
                    .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
                let helpers = {
                    let mut guard = rlm_session.lock().await;
                    if let Err(err) = guard.helpers_add(name.clone(), code) {
                        let value = error_value(
                            "helper_limit_exceeded",
                            err.to_string(),
                            Some("Remove unused helpers and try again"),
                        );
                        return Ok(json_tool_output(value, false));
                    }
                    guard.helpers_list()
                };

                let value = json!({
                    "success": true,
                    "helpers": helpers,
                });
                Ok(json_tool_output(value, true))
            }
            "rlm_helpers_list" => {
                let _: EmptyArgs = parse_arguments(&arguments)?;
                let rlm_session = session
                    .rlm_session()
                    .await
                    .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
                let helpers = {
                    let guard = rlm_session.lock().await;
                    guard.helpers_list()
                };
                let value = json!({
                    "success": true,
                    "helpers": helpers,
                });
                Ok(json_tool_output(value, true))
            }
            "rlm_helpers_remove" => {
                let args: RlmHelpersRemoveArgs = parse_arguments(&arguments)?;
                let name = args.name.trim();
                if name.is_empty() {
                    let value = error_value(
                        "invalid_argument",
                        "name must not be empty",
                        Some("Provide the helper name to remove"),
                    );
                    return Ok(json_tool_output(value, false));
                }

                let rlm_session = session
                    .rlm_session()
                    .await
                    .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
                let (removed, helpers) = {
                    let mut guard = rlm_session.lock().await;
                    let removed = guard
                        .helpers_remove(name)
                        .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
                    (removed, guard.helpers_list())
                };

                let value = json!({
                    "success": true,
                    "removed": removed,
                    "helpers": helpers,
                });
                Ok(json_tool_output(value, true))
            }
            _ => Err(FunctionCallError::RespondToModel(
                "rlm_helpers handler received unsupported tool name".to_string(),
            )),
        }
    }
}
