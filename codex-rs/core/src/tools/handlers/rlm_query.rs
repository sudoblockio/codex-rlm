use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;
use serde_json::json;

use crate::function_tool::FunctionCallError;
use crate::rlm_session::RlmSession;
use crate::rlm_sub_agent::default_sub_agent_tools;
use crate::rlm_sub_agent::run_sub_agent;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::rlm_types::emit_rlm_activity;
use crate::tools::handlers::rlm_types::error_value;
use crate::tools::handlers::rlm_types::json_tool_output;
use crate::tools::registry::ToolHandler;
use crate::tools::registry::ToolKind;

pub(crate) struct RlmQueryHandler;

#[derive(Deserialize)]
struct RlmQueryArgs {
    prompt: String,
    #[serde(default)]
    max_sections: Option<usize>,
    #[serde(default)]
    search_strategy: Option<SearchStrategy>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SearchStrategy {
    Regex,
    Bm25,
    Routing,
    Auto,
}

struct Section {
    content: String,
}

#[async_trait]
impl ToolHandler for RlmQueryHandler {
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
                    "rlm_query handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: RlmQueryArgs = parse_arguments(&arguments)?;
        let prompt = args.prompt.trim();
        if prompt.is_empty() {
            let value = error_value(
                "invalid_argument",
                "prompt must not be empty",
                Some("Provide a prompt to analyze the context"),
            );
            return Ok(json_tool_output(value, false));
        }

        let max_sections = args.max_sections.unwrap_or(10);
        if !(1..=50).contains(&max_sections) {
            let value = error_value(
                "invalid_argument",
                "max_sections must be between 1 and 50",
                Some("Choose a max_sections value between 1 and 50"),
            );
            return Ok(json_tool_output(value, false));
        }

        // Create a short prompt preview for the activity event
        let prompt_preview = if prompt.len() > 50 {
            format!("{}...", &prompt[..50].replace('\n', " "))
        } else {
            prompt.replace('\n', " ")
        };
        emit_rlm_activity(
            &session,
            &turn,
            &call_id,
            "rlm_query",
            &format!("Querying: {prompt_preview}"),
            false,
        )
        .await;

        let rlm_session = session
            .rlm_session()
            .await
            .map_err(|err| FunctionCallError::RespondToModel(err.to_string()))?;

        // Extract sections while holding the lock, then release before sub-agent call
        let (sections, strategy) = {
            let mut guard = rlm_session.lock().await;
            if !guard.has_context() {
                let value = error_value(
                    "context_not_loaded",
                    "rlm_load must be called before rlm_query",
                    Some("Call rlm_load with a path before running rlm_query"),
                );
                return Ok(json_tool_output(value, false));
            }

            let strategy = match args.search_strategy.unwrap_or(SearchStrategy::Auto) {
                SearchStrategy::Auto => {
                    if guard.has_routing() {
                        SearchStrategy::Routing
                    } else {
                        SearchStrategy::Bm25
                    }
                }
                other => other,
            };

            let sections = match strategy {
                SearchStrategy::Regex => match regex_sections(&guard, prompt, max_sections) {
                    Ok(sections) => sections,
                    Err(value) => return Ok(json_tool_output(value, false)),
                },
                SearchStrategy::Bm25 => bm25_sections(&mut guard, prompt, max_sections),
                SearchStrategy::Routing => routing_sections(&guard, prompt, max_sections),
                SearchStrategy::Auto => Vec::new(),
            };

            (sections, strategy)
        }; // Guard released here

        let answer = if sections.is_empty() {
            "No relevant sections found in the loaded context.".to_string()
        } else {
            let composed_prompt = build_query_prompt(prompt, &sections);
            run_sub_agent(
                session.as_ref(),
                turn.as_ref(),
                composed_prompt,
                default_sub_agent_tools(),
            )
            .await?
        };

        // Emit activity complete
        emit_rlm_activity(
            &session,
            &turn,
            &call_id,
            "rlm_query",
            &format!("Analyzed {} sections", sections.len()),
            true,
        )
        .await;

        let budget = {
            let guard = rlm_session.lock().await;
            guard.budget_snapshot()
        };

        let value = json!({
            "success": true,
            "answer": answer,
            "sections_analyzed": sections.len(),
            "search_strategy_used": strategy_string(strategy),
            "budget": budget,
        });
        Ok(json_tool_output(value, true))
    }
}

fn regex_sections(
    session: &RlmSession,
    pattern: &str,
    max_sections: usize,
) -> Result<Vec<Section>, serde_json::Value> {
    let regex = Regex::new(pattern).map_err(|err| {
        error_value(
            "invalid_argument",
            format!("invalid regex: {err}"),
            Some("Provide a valid regex pattern"),
        )
    })?;

    let mut sections = Vec::new();
    let context = session.context();
    for match_ in regex.find_iter(context).take(max_sections) {
        let start = match_.start().saturating_sub(200);
        let end = (match_.end() + 200).min(context.len());
        let snippet = slice_at_boundaries(context, start, end);
        sections.push(Section {
            content: snippet.to_string(),
        });
    }
    Ok(sections)
}

fn bm25_sections(session: &mut RlmSession, prompt: &str, max_sections: usize) -> Vec<Section> {
    session
        .bm25_search(prompt, max_sections)
        .into_iter()
        .map(|result| Section { content: result.text })
        .collect()
}

fn routing_sections(session: &RlmSession, prompt: &str, max_sections: usize) -> Vec<Section> {
    let Some(graph) = session.routing_graph() else {
        return Vec::new();
    };

    let mut sections = Vec::new();
    let mut seen_paths = std::collections::HashSet::new();

    for route in graph.find_routes(prompt).into_iter().take(max_sections * 2) {
        let entry = &route.entry;

        // Skip if we've already included this path
        if seen_paths.contains(&entry.path) {
            continue;
        }

        // Try to get the actual document content
        if let Some(content) = session.document_content(&entry.path) {
            seen_paths.insert(entry.path.clone());

            // Include metadata header followed by actual content
            let header = format!(
                "# {} ({})\n## Path: {}\n## Description: {}\n\n",
                entry.label, route.agents_path, entry.path, entry.description
            );

            // Truncate content if too long (keep first ~4000 chars)
            let truncated_content = if content.len() > 4000 {
                format!("{}...\n[Content truncated]", &content[..4000])
            } else {
                content.to_string()
            };

            sections.push(Section {
                content: format!("{header}{truncated_content}"),
            });

            if sections.len() >= max_sections {
                break;
            }
        }
    }

    // If no content found via routing paths, fall back to metadata only
    if sections.is_empty() {
        for route in graph.find_routes(prompt).into_iter().take(max_sections) {
            let entry = route.entry;
            sections.push(Section {
                content: format!(
                    "Route: {} -> {}\nLabel: {}\nDescription: {}\nScore: {:.2}\n(Document content not found in loaded context)",
                    route.agents_path, entry.path, entry.label, entry.description, route.score
                ),
            });
        }
    }

    sections
}

fn build_query_prompt(prompt: &str, sections: &[Section]) -> String {
    let mut combined = String::from(
        "You are a sub-agent helping with an RLM quick query.\n\
Answer the question using only the provided sections.\n\n",
    );
    combined.push_str(&format!("Question:\n{prompt}\n\n"));
    combined.push_str("Sections:\n");
    for (idx, section) in sections.iter().enumerate() {
        combined.push_str(&format!("--- Section {} ---\n", idx + 1));
        combined.push_str(&section.content);
        combined.push_str("\n\n");
    }
    combined.push_str("Answer:\n");
    combined
}

fn strategy_string(strategy: SearchStrategy) -> &'static str {
    match strategy {
        SearchStrategy::Regex => "regex",
        SearchStrategy::Bm25 => "bm25",
        SearchStrategy::Routing => "routing",
        SearchStrategy::Auto => "auto",
    }
}

fn slice_at_boundaries(content: &str, start: usize, end: usize) -> &str {
    let start = clamp_to_boundary(content, start);
    let end = clamp_to_boundary(content, end);
    content.get(start..end).unwrap_or("")
}

fn clamp_to_boundary(content: &str, mut index: usize) -> usize {
    let max = content.len();
    if index > max {
        index = max;
    }
    while index > 0 && !content.is_char_boundary(index) {
        index -= 1;
    }
    index
}
