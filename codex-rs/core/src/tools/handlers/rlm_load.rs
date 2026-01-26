use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::function_tool::FunctionCallError;
use crate::rlm_session::RlmLoadMode;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::rlm_types::emit_rlm_activity;
use crate::tools::handlers::rlm_types::emit_rlm_status;
use crate::tools::handlers::rlm_types::json_tool_output;
use crate::tools::handlers::rlm_types::validate_path_in_sandbox;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub(crate) struct RlmLoadHandler;

#[derive(Deserialize)]
struct RlmLoadArgs {
    path: String,
}

#[async_trait]
impl ToolHandler for RlmLoadHandler {
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
            call_id,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "rlm_load handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: RlmLoadArgs = parse_arguments(&arguments)?;
        let path = PathBuf::from(&args.path);

        // Validate path is within sandbox boundaries
        if let Err(value) = validate_path_in_sandbox(&path, &turn.sandbox_policy, &turn.cwd) {
            return Ok(json_tool_output(value, false));
        }

        // Emit activity start event
        let path_display = path.display().to_string();
        emit_rlm_activity(
            &session,
            &turn,
            &call_id,
            "rlm_load",
            &format!("Loading files from {path_display}..."),
            false,
        )
        .await;

        let rlm_session = session
            .rlm_session()
            .await
            .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;
        let stats = {
            let mut guard = rlm_session.lock().await;
            guard
                .load_path(&path, RlmLoadMode::Reset)
                .await
                .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?
        };

        // Emit activity complete and status update
        emit_rlm_activity(
            &session,
            &turn,
            &call_id,
            "rlm_load",
            &format!("Loaded {} files", stats.document_count),
            true,
        )
        .await;
        emit_rlm_status(&session, &turn, &rlm_session).await;

        let value = json!({
            "success": true,
            "stats": stats,
        });
        Ok(json_tool_output(value, true))
    }
}
