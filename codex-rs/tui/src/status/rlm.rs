//! RLM status display for the TUI.

use codex_protocol::protocol::RlmStatusSnapshot;
use ratatui::style::Stylize;
use ratatui::text::Line;
use ratatui::text::Span;

use crate::history_cell::CompositeHistoryCell;
use crate::history_cell::HistoryCell;
use crate::history_cell::PlainHistoryCell;

use super::format::FieldFormatter;
use super::helpers::format_tokens_compact;

/// RLM status data for /status card display.
#[derive(Clone, Debug, Default)]
pub(crate) struct StatusRlmData {
    pub context_loaded: bool,
    pub sources: Vec<String>,
    pub document_count: usize,
    pub token_estimate: u64,
    pub memory_keys_count: usize,
    pub memory_bytes_used: usize,
    pub helpers_count: usize,
    pub budget_remaining: u64,
    pub has_routing: bool,
    pub routing_entry_count: usize,
}

impl StatusRlmData {
    pub fn from_snapshot(snapshot: &RlmStatusSnapshot) -> Self {
        Self {
            context_loaded: snapshot.context_loaded,
            sources: snapshot.sources.clone(),
            document_count: snapshot.document_count,
            token_estimate: snapshot.token_estimate,
            memory_keys_count: snapshot.memory_keys.len(),
            memory_bytes_used: snapshot.memory_bytes_used,
            helpers_count: snapshot.helpers.len(),
            budget_remaining: snapshot.budget_remaining_tokens,
            has_routing: snapshot.has_routing,
            routing_entry_count: snapshot.routing_entry_count,
        }
    }

    /// Render the RLM status lines for the status card.
    pub fn render_lines(&self, formatter: &FieldFormatter) -> Vec<Line<'static>> {
        let mut lines = Vec::new();

        if !self.context_loaded {
            lines.push(formatter.line(
                "RLM",
                vec![Span::from("No context loaded").dim()],
            ));
            return lines;
        }

        // Context line: "142 docs (485K tokens)"
        let doc_str = if self.document_count == 1 {
            "1 doc".to_string()
        } else {
            format!("{} docs", self.document_count)
        };
        let token_str = format_tokens_compact(self.token_estimate as i64);
        let context_value = format!("{doc_str} ({token_str} tokens)");
        lines.push(formatter.line("RLM Context", vec![Span::from(context_value)]));

        // Source path(s) on continuation line if present
        if !self.sources.is_empty() {
            let source_display = if self.sources.len() == 1 {
                self.sources[0].clone()
            } else {
                format!("{} sources", self.sources.len())
            };
            lines.push(formatter.continuation(vec![Span::from(source_display).dim()]));
        }

        // Memory line if any keys stored
        if self.memory_keys_count > 0 {
            let key_str = if self.memory_keys_count == 1 {
                "1 key".to_string()
            } else {
                format!("{} keys", self.memory_keys_count)
            };
            let bytes_str = format_bytes(self.memory_bytes_used);
            lines.push(formatter.line(
                "RLM Memory",
                vec![Span::from(format!("{key_str} ({bytes_str})"))],
            ));
        }

        // Helpers line if any loaded
        if self.helpers_count > 0 {
            let helper_str = if self.helpers_count == 1 {
                "1 helper".to_string()
            } else {
                format!("{} helpers", self.helpers_count)
            };
            lines.push(formatter.line("RLM Helpers", vec![Span::from(helper_str)]));
        }

        // Routing line if available
        if self.has_routing {
            let routing_str = format!("{} entries", self.routing_entry_count);
            lines.push(formatter.line("RLM Routing", vec![Span::from(routing_str)]));
        }

        // Budget remaining
        let budget_str = format_tokens_compact(self.budget_remaining as i64);
        lines.push(formatter.line(
            "RLM Budget",
            vec![Span::from(format!("{budget_str} tokens remaining"))],
        ));

        lines
    }

    /// Collect labels for the RLM section.
    pub fn collect_labels(&self, labels: &mut Vec<String>, seen: &mut std::collections::BTreeSet<String>) {
        use super::format::push_label;

        if !self.context_loaded {
            push_label(labels, seen, "RLM");
            return;
        }

        push_label(labels, seen, "RLM Context");
        if self.memory_keys_count > 0 {
            push_label(labels, seen, "RLM Memory");
        }
        if self.helpers_count > 0 {
            push_label(labels, seen, "RLM Helpers");
        }
        if self.has_routing {
            push_label(labels, seen, "RLM Routing");
        }
        push_label(labels, seen, "RLM Budget");
    }
}

/// Format bytes into a human-readable string (e.g., "12.4 KB", "3.2 MB").
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * 1024;

    if bytes < KB {
        format!("{} B", bytes)
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    }
}

/// A simple history cell for RLM status output.
#[derive(Debug)]
pub(crate) struct RlmStatusHistoryCell {
    data: StatusRlmData,
}

impl RlmStatusHistoryCell {
    fn new(data: StatusRlmData) -> Self {
        Self { data }
    }
}

impl HistoryCell for RlmStatusHistoryCell {
    fn display_lines(&self, _width: u16) -> Vec<Line<'static>> {
        // Compute label width for formatting
        let mut labels = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        self.data.collect_labels(&mut labels, &mut seen);
        let formatter = FieldFormatter::from_labels(&labels);

        self.data.render_lines(&formatter)
    }
}

/// Create a dedicated RLM status output for the `/rlm` command.
pub(crate) fn new_rlm_status_output(snapshot: Option<&RlmStatusSnapshot>) -> CompositeHistoryCell {
    let command = PlainHistoryCell::new(vec!["/rlm".magenta().into()]);

    let data = snapshot
        .map(StatusRlmData::from_snapshot)
        .unwrap_or_default();

    let status_cell = RlmStatusHistoryCell::new(data);

    CompositeHistoryCell::new(vec![Box::new(command), Box::new(status_cell)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_works() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(12_700), "12.4 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(3_200_000), "3.1 MB");
    }
}
