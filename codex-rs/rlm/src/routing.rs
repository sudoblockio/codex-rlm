use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingGraph {
    pub manifest_path: String,
    pub entries: Vec<RoutingEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RoutingEntry {
    pub label: String,
    pub path: String,
    pub description: String,
    pub kind: RoutingKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RoutingKind {
    DocsIndex,
    ContextRouting,
    DomainRouter,
    StyleGuide,
    Code,
    Other,
}

impl RoutingKind {
    fn from_path(path: &str) -> Self {
        if path == "docs/AGENTS.md" {
            return Self::DocsIndex;
        }
        if path.ends_with("/docs/context.md") || path == "docs/context.md" {
            return Self::ContextRouting;
        }
        if path.contains("/docs/") && path.ends_with("AGENTS.md") {
            return Self::DomainRouter;
        }
        if path.starts_with("docs/agent/") {
            return Self::StyleGuide;
        }
        if path.starts_with("_") || path.starts_with("tools/") {
            return Self::Code;
        }
        Self::Other
    }
}

#[derive(Clone, Debug)]
pub struct RoutingCache {
    manifest_path: Option<PathBuf>,
    graph: Option<RoutingGraph>,
    summary: Option<String>,
}

impl RoutingCache {
    pub fn empty() -> Self {
        Self {
            manifest_path: None,
            graph: None,
            summary: None,
        }
    }

    pub fn discover(start: &Path, manifest_names: &[String]) -> Result<Self> {
        let manifest = find_manifest(start, manifest_names)?;
        if let Some(path) = manifest {
            return Self::load(&path);
        }
        Ok(Self::empty())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let entries = parse_manifest(&content);
        let graph = RoutingGraph {
            manifest_path: path.to_string_lossy().to_string(),
            entries,
        };
        let summary = summarize_graph(&graph);
        Ok(Self {
            manifest_path: Some(path.to_path_buf()),
            graph: Some(graph),
            summary,
        })
    }

    pub fn graph(&self) -> Option<&RoutingGraph> {
        self.graph.as_ref()
    }

    pub fn summary(&self) -> Option<String> {
        self.summary.clone()
    }

    pub fn manifest_path(&self) -> Option<&Path> {
        self.manifest_path.as_deref()
    }
}

fn find_manifest(start: &Path, manifest_names: &[String]) -> Result<Option<PathBuf>> {
    let mut current = start;
    loop {
        for name in manifest_names {
            let candidate = current.join(name);
            if candidate.is_file() {
                return Ok(Some(candidate));
            }
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }
    Ok(None)
}

fn parse_manifest(content: &str) -> Vec<RoutingEntry> {
    let mut entries = Vec::new();
    let em_dash = '\u{2014}';

    for line in content.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("- [") else {
            continue;
        };
        let Some(label_end) = rest.find("](") else {
            continue;
        };
        let label = rest[..label_end].trim();
        let after_label = &rest[label_end + 2..];
        let Some(path_end) = after_label.find(')') else {
            continue;
        };
        let path = after_label[..path_end].trim();
        let mut description = after_label[path_end + 1..].trim();
        if let Some(stripped) = description.strip_prefix('-') {
            description = stripped.trim();
        } else if let Some(stripped) = description.strip_prefix(em_dash) {
            description = stripped.trim();
        }

        let kind = RoutingKind::from_path(path);
        entries.push(RoutingEntry {
            label: label.to_string(),
            path: path.to_string(),
            description: description.to_string(),
            kind,
        });
    }

    entries
}

fn summarize_graph(graph: &RoutingGraph) -> Option<String> {
    if graph.entries.is_empty() {
        return None;
    }

    let mut docs = None;
    let mut context = None;
    let mut overview = None;
    let mut styles = None;
    let mut tools = None;
    let mut domains = Vec::new();
    let mut included = 0usize;

    for entry in &graph.entries {
        let path = entry.path.as_str();
        if path == "docs/AGENTS.md" {
            docs = Some(path);
            included += 1;
        } else if path.ends_with("docs/context.md") {
            context = Some(path);
            included += 1;
        } else if path.ends_with("docs/overview.md") {
            overview = Some(path);
            included += 1;
        } else if path.contains("docs/agent") {
            styles = Some(path);
            included += 1;
        } else if path.starts_with("tools/") || path.ends_with("tools/AGENTS.md") {
            tools = Some(path);
            included += 1;
        } else if path.contains("/docs/") && path.ends_with("AGENTS.md") {
            domains.push(path);
            included += 1;
        }
    }

    let mut parts = Vec::new();
    if let Some(path) = docs {
        parts.push(format!("docs={path}"));
    }
    if let Some(path) = context {
        parts.push(format!("context={path}"));
    }
    if let Some(path) = overview {
        parts.push(format!("overview={path}"));
    }
    if !domains.is_empty() {
        parts.push(format!("domains={}", domains.join(",")));
    }
    if let Some(path) = styles {
        parts.push(format!("styles={path}"));
    }
    if let Some(path) = tools {
        parts.push(format!("tools={path}"));
    }

    let parts_joined = parts.join(", ");
    let summary = format!("Routing: {parts_joined}");
    truncate_summary(&summary, graph.entries.len(), included)
}

fn truncate_summary(summary: &str, entry_count: usize, included: usize) -> Option<String> {
    let max_len = 300;
    if summary.len() <= max_len {
        return Some(summary.to_string());
    }
    let mut truncated = summary.chars().take(max_len).collect::<String>();
    let omitted = match entry_count.saturating_sub(included) {
        0 => 1,
        value => value,
    };
    truncated.push_str(&format!("...(+{omitted} more)"));
    Some(truncated)
}

pub fn load_routing_graph(path: &Path) -> Result<RoutingGraph> {
    if !path.is_file() {
        let path = path.display();
        bail!("routing manifest not found at {path}");
    }
    let content = fs::read_to_string(path)?;
    Ok(RoutingGraph {
        manifest_path: path.to_string_lossy().to_string(),
        entries: parse_manifest(&content),
    })
}

/// A node in the hierarchical routing graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoutingNode {
    /// Relative path of this AGENTS.md file.
    pub path: String,
    /// Parsed routing entries.
    pub entries: Vec<RoutingEntry>,
    /// Parent AGENTS.md path (if not root).
    pub parent: Option<String>,
    /// Child AGENTS.md paths.
    pub children: Vec<String>,
    /// Depth in the hierarchy (0 = root).
    pub depth: usize,
}

/// Hierarchical routing graph built from all AGENTS.md files.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HierarchicalRoutingGraph {
    /// All routing nodes by their path.
    nodes: HashMap<String, RoutingNode>,
    /// Root AGENTS.md paths (those with no parent).
    roots: Vec<String>,
    /// Total entry count across all nodes.
    total_entries: usize,
}

impl HierarchicalRoutingGraph {
    /// Create an empty hierarchical routing graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a hierarchical routing graph from AGENTS.md files in a DocTreeStore.
    ///
    /// Uses absolute paths for all nodes so that ancestor AGENTS.md files
    /// (discovered above the loaded root) are properly linked as parents
    /// of in-tree AGENTS.md files.
    pub fn from_doc_tree(store: &crate::context::DocTreeStore) -> Self {
        let mut graph = Self::new();
        let root = store
            .root()
            .canonicalize()
            .unwrap_or_else(|_| store.root().to_path_buf());

        // First pass: add ancestor AGENTS.md files (above the loaded root).
        for ancestor in store.ancestor_agents() {
            let key = ancestor.abs_path.to_string_lossy().to_string();
            let entries = parse_manifest(&ancestor.content);
            graph.total_entries += entries.len();
            graph.nodes.insert(
                key.clone(),
                RoutingNode {
                    path: key,
                    entries,
                    parent: None,
                    children: Vec::new(),
                    depth: 0,
                },
            );
        }

        // Second pass: add in-tree AGENTS.md files, converting relative
        // paths to absolute so parent-child walking works across the
        // ancestor/in-tree boundary.
        for rel_path in store.list_agents_files() {
            if let Some(content) = store.document_content(rel_path) {
                let abs = root.join(rel_path);
                let key = abs.to_string_lossy().to_string();
                let entries = parse_manifest(content);
                graph.total_entries += entries.len();
                graph.nodes.insert(
                    key.clone(),
                    RoutingNode {
                        path: key,
                        entries,
                        parent: None,
                        children: Vec::new(),
                        depth: 0,
                    },
                );
            }
        }

        // Third pass: establish parent-child relationships
        let paths: Vec<String> = graph.nodes.keys().cloned().collect();
        for path in &paths {
            let parent_path = find_parent_agents_path(path, &paths);
            if let Some(ref parent) = parent_path {
                if let Some(node) = graph.nodes.get_mut(path) {
                    node.parent = Some(parent.clone());
                }
                if let Some(parent_node) = graph.nodes.get_mut(parent) {
                    parent_node.children.push(path.clone());
                }
            } else {
                graph.roots.push(path.clone());
            }
        }

        // Fourth pass: calculate depths
        for root in &graph.roots.clone() {
            graph.set_depths(root, 0);
        }

        graph
    }

    fn set_depths(&mut self, path: &str, depth: usize) {
        if let Some(node) = self.nodes.get_mut(path) {
            node.depth = depth;
            let children = node.children.clone();
            for child in children {
                self.set_depths(&child, depth + 1);
            }
        }
    }

    /// Get a routing node by path.
    pub fn get_node(&self, path: &str) -> Option<&RoutingNode> {
        self.nodes.get(path)
    }

    /// Get all root nodes.
    pub fn roots(&self) -> &[String] {
        &self.roots
    }

    /// Get total entry count.
    pub fn total_entries(&self) -> usize {
        self.total_entries
    }

    /// Get total node count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Find routes matching a topic query.
    ///
    /// Returns a list of (path, score, entry) tuples for matching entries.
    pub fn find_routes(&self, topic: &str) -> Vec<RouteMatch> {
        let topic_lower = topic.to_lowercase();
        let topic_words: Vec<&str> = topic_lower.split_whitespace().collect();
        let mut matches = Vec::new();

        for (path, node) in &self.nodes {
            for entry in &node.entries {
                let score = calculate_route_score(entry, &topic_lower, &topic_words);
                if score > 0.0 {
                    matches.push(RouteMatch {
                        agents_path: path.clone(),
                        entry: entry.clone(),
                        score,
                        depth: node.depth,
                    });
                }
            }
        }

        // Sort by score descending, then by depth ascending (prefer shallower)
        matches.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.depth.cmp(&b.depth))
        });

        matches
    }

    /// Get the path from root to a given AGENTS.md file.
    pub fn path_to_node(&self, path: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = path;

        while let Some(node) = self.nodes.get(current) {
            result.push(current.to_string());
            if let Some(ref parent) = node.parent {
                current = parent;
            } else {
                break;
            }
        }

        result.reverse();
        result
    }

    /// List all documents referenced from a given AGENTS.md and its descendants.
    pub fn list_reachable_docs(&self, from: &str) -> Vec<String> {
        let mut docs = Vec::new();
        self.collect_reachable_docs(from, &mut docs);
        docs
    }

    fn collect_reachable_docs(&self, path: &str, docs: &mut Vec<String>) {
        if let Some(node) = self.nodes.get(path) {
            // Add direct doc references
            for entry in &node.entries {
                if !entry.path.ends_with("AGENTS.md") {
                    docs.push(entry.path.clone());
                }
            }
            // Recurse into children
            for child in &node.children {
                self.collect_reachable_docs(child, docs);
            }
        }
    }
}

/// A route match result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouteMatch {
    /// Path to the AGENTS.md containing this entry.
    pub agents_path: String,
    /// The matching routing entry.
    pub entry: RoutingEntry,
    /// Match score (higher is better).
    pub score: f64,
    /// Depth in the routing hierarchy.
    pub depth: usize,
}

/// Find the parent AGENTS.md path for a given AGENTS.md path.
fn find_parent_agents_path(path: &str, all_paths: &[String]) -> Option<String> {
    // Get the directory containing this AGENTS.md
    let path_dir = Path::new(path).parent()?;

    // Walk up the directory tree looking for another AGENTS.md
    let mut current = path_dir.parent();
    while let Some(dir) = current {
        let candidate = dir.join("AGENTS.md");
        let candidate_str = candidate.to_string_lossy().to_string();
        if all_paths.contains(&candidate_str) {
            return Some(candidate_str);
        }
        current = dir.parent();
    }

    // Check for root AGENTS.md
    if all_paths.contains(&"AGENTS.md".to_string()) && path != "AGENTS.md" {
        return Some("AGENTS.md".to_string());
    }

    None
}

/// Calculate a match score for a routing entry against a topic query.
fn calculate_route_score(entry: &RoutingEntry, topic_lower: &str, topic_words: &[&str]) -> f64 {
    let mut score = 0.0;

    let label_lower = entry.label.to_lowercase();
    let desc_lower = entry.description.to_lowercase();
    let path_lower = entry.path.to_lowercase();

    // Exact phrase match in label (highest value)
    if label_lower.contains(topic_lower) {
        score += 10.0;
    }

    // Exact phrase match in description
    if desc_lower.contains(topic_lower) {
        score += 5.0;
    }

    // Exact phrase match in path
    if path_lower.contains(topic_lower) {
        score += 3.0;
    }

    // Word matches
    for word in topic_words {
        if word.len() < 2 {
            continue;
        }
        if label_lower.contains(word) {
            score += 2.0;
        }
        if desc_lower.contains(word) {
            score += 1.0;
        }
        if path_lower.contains(word) {
            score += 0.5;
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn loads_routing_graph_and_summary() {
        let content = r#"
- [Documentation standards](docs/docs.md) — Normative design-doc specification
- [Docs index](docs/AGENTS.md) — Documentation entry points and references
- [Context routing](docs/context.md) — Tier-0 routing rules
- [Platform docs](_platform/docs/AGENTS.md) — Platform architecture and services
"#;
        let file = tempfile::NamedTempFile::new().unwrap();
        fs::write(file.path(), content).unwrap();

        let cache = RoutingCache::load(file.path()).unwrap();
        let graph = cache.graph().unwrap();
        assert_eq!(graph.entries.len(), 4);

        let docs_entry = graph
            .entries
            .iter()
            .find(|entry| entry.path == "docs/AGENTS.md")
            .unwrap();
        assert_eq!(docs_entry.kind, RoutingKind::DocsIndex);

        let context_entry = graph
            .entries
            .iter()
            .find(|entry| entry.path == "docs/context.md")
            .unwrap();
        assert_eq!(context_entry.kind, RoutingKind::ContextRouting);

        let platform_entry = graph
            .entries
            .iter()
            .find(|entry| entry.path == "_platform/docs/AGENTS.md")
            .unwrap();
        assert_eq!(platform_entry.kind, RoutingKind::DomainRouter);

        let summary = cache.summary().unwrap();
        assert!(summary.contains("Routing:"));
        assert!(summary.contains("docs=docs/AGENTS.md"));
        assert!(summary.contains("context=docs/context.md"));
    }

    #[test]
    fn find_routes_matches_by_topic() {
        let entry1 = RoutingEntry {
            label: "Release Process".to_string(),
            path: "docs/release.md".to_string(),
            description: "How to release new versions".to_string(),
            kind: RoutingKind::Other,
        };
        let entry2 = RoutingEntry {
            label: "Authentication".to_string(),
            path: "docs/auth.md".to_string(),
            description: "User authentication and authorization".to_string(),
            kind: RoutingKind::Other,
        };

        let mut nodes = HashMap::new();
        nodes.insert(
            "AGENTS.md".to_string(),
            RoutingNode {
                path: "AGENTS.md".to_string(),
                entries: vec![entry1, entry2],
                parent: None,
                children: vec![],
                depth: 0,
            },
        );

        let graph = HierarchicalRoutingGraph {
            nodes,
            roots: vec!["AGENTS.md".to_string()],
            total_entries: 2,
        };

        // Search for "release"
        let matches = graph.find_routes("release");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].entry.label, "Release Process");
        assert!(matches[0].score > 0.0);

        // Search for "auth"
        let auth_matches = graph.find_routes("auth");
        assert_eq!(auth_matches.len(), 1);
        assert_eq!(auth_matches[0].entry.label, "Authentication");
    }

    #[test]
    fn hierarchical_graph_parent_child() {
        // Manually test the parent finding logic with relative paths
        let paths = vec![
            "AGENTS.md".to_string(),
            "docs/AGENTS.md".to_string(),
            "docs/api/AGENTS.md".to_string(),
        ];

        // docs/api/AGENTS.md should have docs/AGENTS.md as parent
        let parent = find_parent_agents_path("docs/api/AGENTS.md", &paths);
        assert_eq!(parent, Some("docs/AGENTS.md".to_string()));

        // docs/AGENTS.md should have root AGENTS.md as parent
        let parent2 = find_parent_agents_path("docs/AGENTS.md", &paths);
        assert_eq!(parent2, Some("AGENTS.md".to_string()));

        // Root AGENTS.md has no parent
        let parent3 = find_parent_agents_path("AGENTS.md", &paths);
        assert_eq!(parent3, None);
    }

    #[test]
    fn hierarchical_graph_parent_child_absolute_paths() {
        // Test parent finding with absolute paths (as used when ancestors are present)
        let paths = vec![
            "/Users/project/AGENTS.md".to_string(),
            "/Users/project/src/AGENTS.md".to_string(),
            "/Users/project/src/docs/AGENTS.md".to_string(),
        ];

        // src/docs/AGENTS.md should have src/AGENTS.md as parent
        let parent =
            find_parent_agents_path("/Users/project/src/docs/AGENTS.md", &paths);
        assert_eq!(parent, Some("/Users/project/src/AGENTS.md".to_string()));

        // src/AGENTS.md should have project root AGENTS.md as parent
        let parent2 =
            find_parent_agents_path("/Users/project/src/AGENTS.md", &paths);
        assert_eq!(parent2, Some("/Users/project/AGENTS.md".to_string()));

        // Root AGENTS.md has no parent
        let parent3 =
            find_parent_agents_path("/Users/project/AGENTS.md", &paths);
        assert_eq!(parent3, None);
    }

    #[tokio::test]
    async fn from_doc_tree_includes_ancestor_agents() {
        use crate::context::{ContextSource, ContextStore, DocTreeStore};

        // Create a directory structure:
        //   /tmp_dir/AGENTS.md          (ancestor - above root)
        //   /tmp_dir/project/           (this is the "root" we load)
        //   /tmp_dir/project/AGENTS.md  (in-tree root)
        //   /tmp_dir/project/src/AGENTS.md (in-tree child)
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path();

        // Ancestor AGENTS.md (above the loaded root)
        fs::write(
            base.join("AGENTS.md"),
            "- [Project](project/AGENTS.md) - Project root docs\n",
        )
        .unwrap();

        // Project root (this is what we'll load)
        let project = base.join("project");
        std::fs::create_dir_all(project.join("src")).unwrap();

        fs::write(
            project.join("AGENTS.md"),
            "- [Source](src/AGENTS.md) - Source code docs\n",
        )
        .unwrap();

        fs::write(
            project.join("src/AGENTS.md"),
            "- [Main](main.rs) - Entry point\n",
        )
        .unwrap();

        fs::write(project.join("src/main.rs"), "fn main() {}").unwrap();

        // Load from project/ (not the base)
        let mut store = DocTreeStore::new();
        store
            .load(ContextSource::DocTree(project.clone()))
            .await
            .unwrap();

        // Should have discovered the ancestor
        assert_eq!(
            store.ancestor_agents().len(),
            1,
            "should find one ancestor AGENTS.md above the loaded root"
        );

        // Build routing graph
        let graph = HierarchicalRoutingGraph::from_doc_tree(&store);

        // Should have 3 nodes: ancestor + 2 in-tree
        assert_eq!(graph.node_count(), 3);

        // Should have exactly 1 root (the ancestor)
        assert_eq!(graph.roots().len(), 1);

        // The in-tree root AGENTS.md should have the ancestor as parent
        let canon_project = project.canonicalize().unwrap();
        let in_tree_root_key = canon_project.join("AGENTS.md");
        let in_tree_root_node = graph
            .get_node(&in_tree_root_key.to_string_lossy())
            .expect("in-tree root should be in graph");
        assert!(
            in_tree_root_node.parent.is_some(),
            "in-tree root should have ancestor as parent"
        );

        // The src/AGENTS.md should have the in-tree root as parent
        let src_key = canon_project.join("src/AGENTS.md");
        let src_node = graph
            .get_node(&src_key.to_string_lossy())
            .expect("src AGENTS.md should be in graph");
        assert_eq!(
            src_node.parent.as_deref(),
            Some(in_tree_root_key.to_string_lossy().as_ref()),
            "src/AGENTS.md should have project/AGENTS.md as parent"
        );
    }

    #[test]
    fn route_score_calculation() {
        let entry = RoutingEntry {
            label: "API Reference".to_string(),
            path: "docs/api/reference.md".to_string(),
            description: "Complete API documentation for developers".to_string(),
            kind: RoutingKind::Other,
        };

        // Exact phrase in label should score highest
        let score1 = calculate_route_score(&entry, "api reference", &["api", "reference"]);
        let score2 = calculate_route_score(&entry, "documentation", &["documentation"]);
        let score3 = calculate_route_score(&entry, "unrelated", &["unrelated"]);

        assert!(
            score1 > score2,
            "Label match should score higher than description match"
        );
        assert!(
            score2 > score3,
            "Description match should score higher than no match"
        );
        assert_eq!(score3, 0.0, "No match should have zero score");
    }
}
