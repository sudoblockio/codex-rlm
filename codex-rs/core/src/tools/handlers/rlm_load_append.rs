use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::function_tool::FunctionCallError;
use crate::rlm_session::RlmLoadMode;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::rlm_types::json_tool_output;
use crate::tools::handlers::rlm_types::validate_existing_absolute_path;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub(crate) struct RlmLoadAppendHandler;

#[derive(Deserialize)]
struct RlmLoadAppendArgs {
    path: String,
}

#[async_trait]
impl ToolHandler for RlmLoadAppendHandler {
    fn kind(&self) -> ToolKind {
        ToolKind::Function
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<crate::tools::context::ToolOutput, FunctionCallError> {
        let ToolInvocation {
            payload, session, ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "rlm_load_append handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: RlmLoadAppendArgs = parse_arguments(&arguments)?;
        let path = PathBuf::from(&args.path);
        if let Err(value) = validate_existing_absolute_path(&path) {
            return Ok(json_tool_output(value, false));
        }

        let rlm_session = session
            .rlm_session()
            .await
            .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
        let stats = {
            let mut guard = rlm_session.lock().await;
            guard
                .load_path(&path, RlmLoadMode::Append)
                .await
                .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?
        };

        let value = json!({
            "success": true,
            "stats": stats,
        });
        Ok(json_tool_output(value, true))
    }
}
