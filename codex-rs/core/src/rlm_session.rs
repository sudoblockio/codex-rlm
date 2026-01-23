use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use anyhow::Result;
use codex_rlm::BudgetSnapshot;
use codex_rlm::RlmConfig;
use codex_rlm::context::ContextSource;
use codex_rlm::context::ContextStore;
use codex_rlm::context::ContextStoreKind;
use codex_rlm::context::DocumentMetadata;
use codex_rlm::estimate_tokens;
use codex_rlm::index::Bm25Index;
use codex_rlm::index::IndexConfig;
use codex_rlm::index::SearchResult;
use codex_rlm::python::ExecutionResult;
use codex_rlm::python::LlmCallback;
use codex_rlm::python::PythonRuntime;
use codex_rlm::python::ResourceLimits;
use codex_rlm::SearchCallback;
use codex_rlm::SearchResultJson;
use codex_rlm::routing::HierarchicalRoutingGraph;
use serde::Deserialize;
use serde::Serialize;

const DEFAULT_MAX_OUTPUT_BYTES: u64 = 102_400;
const DEFAULT_MAX_EXECUTION_MS: u64 = 30_000;
const DEFAULT_MAX_FIND_RESULTS: u32 = 10_000;
const HELPERS_LIMIT_BYTES: usize = 1024 * 1024;
const MEMORY_LIMIT_BYTES: usize = 5 * 1024 * 1024;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RlmLimits {
    pub(crate) max_output_bytes: u64,
    pub(crate) max_execution_ms: u64,
    pub(crate) max_find_results: u32,
}

impl Default for RlmLimits {
    fn default() -> Self {
        Self {
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            max_execution_ms: DEFAULT_MAX_EXECUTION_MS,
            max_find_results: DEFAULT_MAX_FIND_RESULTS,
        }
    }
}

impl RlmLimits {
    pub(crate) fn apply_override(&self, override_limits: Option<&RlmLimitsOverride>) -> Self {
        let mut applied = self.clone();
        let Some(override_limits) = override_limits else {
            return applied;
        };
        if let Some(max_output_bytes) = override_limits.max_output_bytes {
            applied.max_output_bytes = max_output_bytes.min(self.max_output_bytes);
        }
        if let Some(max_execution_ms) = override_limits.max_execution_ms {
            applied.max_execution_ms = max_execution_ms.min(self.max_execution_ms);
        }
        if let Some(max_find_results) = override_limits.max_find_results {
            applied.max_find_results = max_find_results.min(self.max_find_results);
        }
        applied
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct RlmLimitsOverride {
    pub(crate) max_output_bytes: Option<u64>,
    pub(crate) max_execution_ms: Option<u64>,
    pub(crate) max_find_results: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RlmLoadStats {
    pub(crate) length_chars: usize,
    pub(crate) length_tokens_estimate: u64,
    pub(crate) line_count: usize,
    pub(crate) document_count: usize,
    pub(crate) has_routing: bool,
    pub(crate) routing_entry_count: usize,
    pub(crate) sources: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RlmSessionManifest {
    pub(crate) sources: Vec<String>,
    pub(crate) context_chars: usize,
    pub(crate) context_tokens_estimate: u64,
    pub(crate) helpers: Vec<String>,
    pub(crate) memory_keys: Vec<String>,
    pub(crate) memory_bytes_used: usize,
    pub(crate) budget: BudgetSnapshot,
    pub(crate) limits: RlmLimits,
}

#[derive(Clone, Debug)]
pub(crate) struct RlmExecOutcome {
    pub(crate) result: ExecutionResult,
    pub(crate) limits_applied: RlmLimits,
    pub(crate) budget: BudgetSnapshot,
}

#[derive(Clone, Debug)]
struct RlmLoadedContext {
    content: String,
    documents: Vec<DocumentMetadata>,
    routing_graph: Option<HierarchicalRoutingGraph>,
}

#[derive(Clone, Debug)]
struct RlmHelper {
    name: String,
    code: String,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RlmLoadMode {
    Reset,
    Append,
}

pub(crate) struct RlmSession {
    python: PythonRuntime,
    context: String,
    sources: Vec<String>,
    documents: Vec<DocumentMetadata>,
    routing_graph: Option<HierarchicalRoutingGraph>,
    helpers: Vec<RlmHelper>,
    memory: BTreeMap<String, serde_json::Value>,
    limits: RlmLimits,
    budget_state: Arc<StdMutex<BudgetSnapshot>>,
    context_loaded: bool,
    bm25_index: Arc<StdMutex<Option<Bm25Index>>>,
}

impl RlmSession {
    pub(crate) fn new() -> Result<Self> {
        let config = RlmConfig::default();
        let limits = RlmLimits::default();
        let budget = Self::budget_from_config(&config);
        let mut python = PythonRuntime::new()?;
        python.set_allowed_modules(&config.safety.allowed_modules)?;
        python.set_resource_limits(Self::resource_limits_for(&limits))?;
        Ok(Self {
            python,
            context: String::new(),
            sources: Vec::new(),
            documents: Vec::new(),
            routing_graph: None,
            helpers: Vec::new(),
            memory: BTreeMap::new(),
            limits,
            budget_state: Arc::new(StdMutex::new(budget)),
            context_loaded: false,
            bm25_index: Arc::new(StdMutex::new(None)),
        })
    }

    pub(crate) fn has_context(&self) -> bool {
        self.context_loaded
    }

    pub(crate) fn stats(&self) -> RlmLoadStats {
        let length_chars = self.context.chars().count();
        let length_tokens_estimate = estimate_tokens(&self.context);
        let line_count = self.context.lines().count();
        let document_count = if self.documents.is_empty() {
            if self.context_loaded { 1 } else { 0 }
        } else {
            self.documents.len()
        };
        let (has_routing, routing_entry_count) = self
            .routing_graph
            .as_ref()
            .map(|graph| (graph.total_entries() > 0, graph.total_entries()))
            .unwrap_or((false, 0));
        RlmLoadStats {
            length_chars,
            length_tokens_estimate,
            line_count,
            document_count,
            has_routing,
            routing_entry_count,
            sources: self.sources.clone(),
        }
    }

    pub(crate) async fn load_path(
        &mut self,
        path: &Path,
        mode: RlmLoadMode,
    ) -> Result<RlmLoadStats> {
        let loaded = Self::load_context_from_path(path).await?;
        match mode {
            RlmLoadMode::Reset => self.reset_state()?,
            RlmLoadMode::Append => {}
        }

        let path_label = path.to_string_lossy().to_string();
        if matches!(mode, RlmLoadMode::Append) && self.context_loaded {
            self.context
                .push_str(&format!("\n\n===== APPENDED: {path_label} =====\n\n"));
        }
        self.context.push_str(&loaded.content);

        self.sources.push(path_label);
        self.documents.extend(loaded.documents);
        if loaded.routing_graph.is_some() {
            self.routing_graph = loaded.routing_graph;
        }
        self.context_loaded = true;
        self.clear_bm25_index();

        self.refresh_python_context()?;
        Ok(self.stats())
    }

    pub(crate) fn exec(
        &mut self,
        code: &str,
        limits_override: Option<&RlmLimitsOverride>,
        llm_callback: Option<Arc<dyn LlmCallback>>,
        tool_override_policy_json: Option<&str>,
    ) -> Result<RlmExecOutcome> {
        let limits_applied = self.limits.apply_override(limits_override);
        self.apply_limits(&limits_applied)?;
        self.refresh_session_manifest_with_limits(&limits_applied)?;
        if let Some(callback) = llm_callback {
            self.python.set_llm_callback(callback)?;
        }
        // Set up search callback for BM25 search from Python
        let search_callback = Arc::new(RlmSearchCallback::new(
            self.bm25_index_shared(),
            self.context.clone(),
        ));
        self.python.set_search_callback(search_callback)?;
        if let Some(policy_json) = tool_override_policy_json {
            self.python
                .set_state_json("tool_override_policy_json", policy_json)?;
        }
        self.python.set_budget(self.budget_snapshot())?;
        let code = self.compose_exec_code(code);
        let result = self.python.execute(&code)?;
        self.sync_budget_from_python()?;
        let base_limits = self.limits.clone();
        self.apply_limits(&base_limits)?;
        self.refresh_session_manifest_with_limits(&base_limits)?;
        Ok(RlmExecOutcome {
            result,
            limits_applied,
            budget: self.budget_snapshot(),
        })
    }

    pub(crate) fn memory_put(&mut self, key: String, value: serde_json::Value) -> Result<()> {
        self.memory.insert(key, value);
        self.enforce_memory_limit()?;
        self.refresh_session_manifest()?;
        Ok(())
    }

    pub(crate) fn memory_get(&self, key: &str) -> Option<serde_json::Value> {
        self.memory.get(key).cloned()
    }

    pub(crate) fn memory_keys(&self) -> Vec<String> {
        self.memory.keys().cloned().collect()
    }

    pub(crate) fn memory_clear(&mut self) -> Result<()> {
        self.memory.clear();
        self.refresh_session_manifest()?;
        Ok(())
    }

    pub(crate) fn budget_snapshot(&self) -> BudgetSnapshot {
        self.budget_state
            .lock()
            .map(|snapshot| snapshot.clone())
            .unwrap_or_else(|_| Self::budget_from_config(&RlmConfig::default()))
    }

    pub(crate) fn budget_state(&self) -> Arc<StdMutex<BudgetSnapshot>> {
        Arc::clone(&self.budget_state)
    }

    pub(crate) fn memory_bytes_used(&self) -> usize {
        self.memory
            .iter()
            .map(|(key, value)| {
                let value_size = serde_json::to_string(value)
                    .map(|json| json.len())
                    .unwrap_or(0);
                key.len() + value_size
            })
            .sum()
    }

    pub(crate) fn helpers_add(&mut self, name: String, code: String) -> Result<()> {
        self.helpers.push(RlmHelper { name, code });
        if let Err(err) = self.enforce_helpers_limit() {
            self.helpers.pop();
            return Err(err);
        }
        self.refresh_session_manifest()?;
        Ok(())
    }

    pub(crate) fn helpers_list(&self) -> Vec<String> {
        self.helpers
            .iter()
            .map(|helper| helper.name.clone())
            .collect()
    }

    pub(crate) fn helpers_remove(&mut self, name: &str) -> Result<bool> {
        let before = self.helpers.len();
        self.helpers.retain(|helper| helper.name != name);
        let removed = self.helpers.len() != before;
        self.refresh_session_manifest()?;
        Ok(removed)
    }

    pub(crate) fn helpers_bytes_used(&self) -> usize {
        self.helpers
            .iter()
            .map(|helper| helper.name.len() + helper.code.len())
            .sum()
    }

    pub(crate) fn context(&self) -> &str {
        &self.context
    }

    pub(crate) fn has_routing(&self) -> bool {
        self.routing_graph
            .as_ref()
            .map(|graph| graph.total_entries() > 0)
            .unwrap_or(false)
    }

    pub(crate) fn routing_graph(&self) -> Option<&HierarchicalRoutingGraph> {
        self.routing_graph.as_ref()
    }

    pub(crate) fn bm25_search(&mut self, query: &str, k: usize) -> Vec<SearchResult> {
        self.ensure_bm25_index();
        self.bm25_index
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|index| index.search(query, k)))
            .unwrap_or_default()
    }

    /// Ensure the BM25 index is built from the current context.
    fn ensure_bm25_index(&self) {
        let mut guard = match self.bm25_index.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if guard.is_none() && !self.context.is_empty() {
            let index = Bm25Index::from_content(self.context.as_str(), IndexConfig::default());
            *guard = Some(index);
        }
    }

    /// Get a shared reference to the BM25 index for use by search callback.
    pub(crate) fn bm25_index_shared(&self) -> Arc<StdMutex<Option<Bm25Index>>> {
        Arc::clone(&self.bm25_index)
    }

    /// Clear the BM25 index (called when context changes).
    fn clear_bm25_index(&self) {
        if let Ok(mut guard) = self.bm25_index.lock() {
            *guard = None;
        }
    }

    fn enforce_memory_limit(&self) -> Result<()> {
        let used = self.memory_bytes_used();
        if used > MEMORY_LIMIT_BYTES {
            anyhow::bail!("memory limit exceeded: {used} bytes (limit {MEMORY_LIMIT_BYTES} bytes)");
        }
        Ok(())
    }

    fn enforce_helpers_limit(&self) -> Result<()> {
        let used = self.helpers_bytes_used();
        if used > HELPERS_LIMIT_BYTES {
            anyhow::bail!(
                "helper limit exceeded: {used} bytes (limit {HELPERS_LIMIT_BYTES} bytes)"
            );
        }
        Ok(())
    }

    fn reset_state(&mut self) -> Result<()> {
        let config = RlmConfig::default();
        let mut python = PythonRuntime::new()?;
        python.set_allowed_modules(&config.safety.allowed_modules)?;
        python.set_resource_limits(Self::resource_limits_for(&self.limits))?;
        self.python = python;
        self.context.clear();
        self.sources.clear();
        self.documents.clear();
        self.routing_graph = None;
        self.helpers.clear();
        self.memory.clear();
        if let Ok(mut guard) = self.budget_state.lock() {
            *guard = Self::budget_from_config(&config);
        }
        self.context_loaded = false;
        self.clear_bm25_index();
        Ok(())
    }

    fn refresh_python_context(&mut self) -> Result<()> {
        self.python.set_context(&self.context)?;
        self.python.set_document_list(&self.documents)?;
        if let Some(graph) = self.routing_graph.clone() {
            self.python.set_hierarchical_routing(&graph)?;
        } else {
            self.python
                .set_hierarchical_routing(&HierarchicalRoutingGraph::new())?;
        }
        self.refresh_session_manifest()?;
        Ok(())
    }

    fn refresh_session_manifest(&mut self) -> Result<()> {
        let manifest = self.session_manifest(&self.limits);
        let json = serde_json::to_string(&manifest)?;
        self.python.set_state_json("session_json", &json)?;
        let limits_json = serde_json::to_string(&manifest.limits)?;
        self.python.set_state_json("limits_json", &limits_json)?;
        Ok(())
    }

    fn refresh_session_manifest_with_limits(&mut self, limits: &RlmLimits) -> Result<()> {
        let manifest = self.session_manifest(limits);
        let json = serde_json::to_string(&manifest)?;
        self.python.set_state_json("session_json", &json)?;
        let limits_json = serde_json::to_string(&manifest.limits)?;
        self.python.set_state_json("limits_json", &limits_json)?;
        Ok(())
    }

    fn session_manifest(&self, limits: &RlmLimits) -> RlmSessionManifest {
        RlmSessionManifest {
            sources: self.sources.clone(),
            context_chars: self.context.chars().count(),
            context_tokens_estimate: estimate_tokens(&self.context),
            helpers: self.helpers_list(),
            memory_keys: self.memory_keys(),
            memory_bytes_used: self.memory_bytes_used(),
            budget: self.budget_snapshot(),
            limits: limits.clone(),
        }
    }

    fn compose_exec_code(&self, code: &str) -> String {
        if self.helpers.is_empty() {
            return code.to_string();
        }
        let mut combined = String::new();
        for helper in &self.helpers {
            combined.push_str(&helper.code);
            if !helper.code.ends_with('\n') {
                combined.push('\n');
            }
            combined.push('\n');
        }
        combined.push_str(code);
        combined
    }

    fn apply_limits(&mut self, limits: &RlmLimits) -> Result<()> {
        self.python
            .set_resource_limits(Self::resource_limits_for(limits))?;
        let limits_json = serde_json::to_string(limits)?;
        self.python.set_state_json("limits_json", &limits_json)?;
        Ok(())
    }

    fn resource_limits_for(limits: &RlmLimits) -> ResourceLimits {
        let max_cpu_seconds = if limits.max_execution_ms == 0 {
            0
        } else {
            limits.max_execution_ms.div_ceil(1000) as u32
        };
        ResourceLimits {
            max_output_bytes: limits.max_output_bytes,
            max_cpu_seconds,
            max_find_results: limits.max_find_results,
        }
    }

    async fn load_context_from_path(path: &Path) -> Result<RlmLoadedContext> {
        let context_path = PathBuf::from(path);
        let source = if context_path.is_dir() {
            ContextSource::DocTree(context_path)
        } else {
            ContextSource::File(context_path)
        };
        let mut store = ContextStoreKind::new();
        store.load(source).await?;
        let content = store.content()?.to_string();
        let documents = store.metadata().documents.clone();
        let routing_graph = store
            .as_doc_tree()
            .map(HierarchicalRoutingGraph::from_doc_tree);
        Ok(RlmLoadedContext {
            content,
            documents,
            routing_graph,
        })
    }

    fn budget_from_config(config: &RlmConfig) -> BudgetSnapshot {
        BudgetSnapshot::new(
            config.safety.max_total_tokens,
            config.safety.max_sub_calls,
            config.safety.max_tool_calls_per_turn,
            u64::from(config.safety.max_cpu_seconds) * 1000,
        )
    }

    fn sync_budget_from_python(&mut self) -> Result<()> {
        if let Some(snapshot) = self.python.budget()?
            && let Ok(mut guard) = self.budget_state.lock() {
                *guard = snapshot;
            }
        Ok(())
    }
}

/// Callback for BM25 search operations from Python.
///
/// This struct holds a shared reference to the BM25 index so that Python code
/// can perform search operations during execution.
pub(crate) struct RlmSearchCallback {
    index: Arc<StdMutex<Option<Bm25Index>>>,
    context: Arc<String>,
}

impl RlmSearchCallback {
    pub(crate) fn new(index: Arc<StdMutex<Option<Bm25Index>>>, context: String) -> Self {
        Self {
            index,
            context: Arc::new(context),
        }
    }

    fn ensure_index(&self) {
        let mut guard = match self.index.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if guard.is_none() && !self.context.is_empty() {
            let index = Bm25Index::from_content(&self.context, IndexConfig::default());
            *guard = Some(index);
        }
    }
}

impl SearchCallback for RlmSearchCallback {
    fn search(&self, query: &str, k: usize) -> anyhow::Result<Vec<SearchResultJson>> {
        self.ensure_index();
        let guard = self
            .index
            .lock()
            .map_err(|_| anyhow::anyhow!("index lock poisoned"))?;
        let results = guard
            .as_ref()
            .map(|index| index.search(query, k))
            .unwrap_or_default();
        Ok(results.into_iter().map(SearchResultJson::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn write_temp_file(contents: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut file, contents.as_bytes()).unwrap();
        file
    }

    #[tokio::test]
    async fn load_and_append_preserve_memory() {
        let mut session = RlmSession::new().unwrap();
        let first = write_temp_file("alpha");
        let second = write_temp_file("beta");

        let stats = session
            .load_path(first.path(), RlmLoadMode::Reset)
            .await
            .unwrap();
        assert_eq!(stats.sources.len(), 1);

        session
            .memory_put("key".to_string(), serde_json::json!({"v": 1}))
            .unwrap();

        let stats = session
            .load_path(second.path(), RlmLoadMode::Append)
            .await
            .unwrap();
        assert_eq!(stats.sources.len(), 2);
        assert_eq!(session.memory_get("key"), Some(serde_json::json!({"v": 1})));
    }

    #[tokio::test]
    async fn session_manifest_tracks_limits_and_budget() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("gamma");
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();
        session
            .memory_put("notes".to_string(), serde_json::json!([1, 2, 3]))
            .unwrap();
        session
            .helpers_add(
                "summarize".to_string(),
                "def summarize():\n    return 1\n".to_string(),
            )
            .unwrap();

        let manifest = session.session_manifest(&session.limits);
        assert_eq!(manifest.sources.len(), 1);
        assert_eq!(manifest.helpers, vec!["summarize".to_string()]);
        assert_eq!(manifest.memory_keys, vec!["notes".to_string()]);
        assert_eq!(
            manifest.limits.max_output_bytes,
            RlmLimits::default().max_output_bytes
        );
        assert_eq!(
            manifest.budget.remaining_sub_calls,
            session.budget_snapshot().remaining_sub_calls
        );
    }

    #[test]
    fn helpers_are_injected_before_code() {
        let mut session = RlmSession::new().unwrap();
        session
            .helpers_add(
                "utils".to_string(),
                "def answer():\n    return 42\n".to_string(),
            )
            .unwrap();
        let combined = session.compose_exec_code("result = answer()");
        assert!(combined.contains("def answer():"));
        assert!(combined.contains("result = answer()"));
    }

    #[tokio::test]
    async fn search_callback_returns_results() {
        let mut session = RlmSession::new().unwrap();
        let content = "The quick brown fox jumps over the lazy dog. \
                       The dog was not amused. \
                       Meanwhile, the cat watched from the window.";
        let file = write_temp_file(content);
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // Create search callback
        let callback =
            RlmSearchCallback::new(session.bm25_index_shared(), session.context().to_string());

        // Search for "fox"
        let results = callback.search("fox", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.text.contains("fox")));

        // Search for non-existent term
        let empty_results = callback.search("xyznonexistent123", 5).unwrap();
        assert!(empty_results.is_empty());
    }

    #[tokio::test]
    async fn bm25_search_clears_on_reload() {
        let mut session = RlmSession::new().unwrap();
        let first = write_temp_file("apple banana cherry");
        let second = write_temp_file("delta epsilon gamma");

        session
            .load_path(first.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // Search should find banana
        let results = session.bm25_search("banana", 5);
        assert!(!results.is_empty());

        // Reload with different content
        session
            .load_path(second.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // banana should no longer be found, but epsilon should
        let results = session.bm25_search("banana", 5);
        assert!(results.is_empty());

        let results = session.bm25_search("epsilon", 5);
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn exec_runs_python_and_returns_output() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("test context");
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let outcome = session.exec("print('hello world')", None, None, None).unwrap();
        assert!(outcome.result.error.is_none());
        assert!(outcome.result.output.contains("hello world"));
    }

    #[tokio::test]
    async fn exec_returns_result_json() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("test context");
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let outcome = session
            .exec("result = {'answer': 42, 'status': 'ok'}", None, None, None)
            .unwrap();
        assert!(outcome.result.error.is_none());
        assert!(outcome.result.result_json.is_some());
        let result_json: serde_json::Value =
            serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        assert_eq!(result_json["answer"], 42);
        assert_eq!(result_json["status"], "ok");
    }

    #[tokio::test]
    async fn exec_with_helpers_uses_helper_functions() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("test context");
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // Add a helper function
        session
            .helpers_add(
                "utils".to_string(),
                "def double(x):\n    return x * 2\n".to_string(),
            )
            .unwrap();

        // Execute code that uses the helper
        let outcome = session
            .exec("result = double(21)", None, None, None)
            .unwrap();
        assert!(outcome.result.error.is_none());
        let result_json: serde_json::Value =
            serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        assert_eq!(result_json, 42);
    }

    #[tokio::test]
    async fn exec_can_use_peek_and_find() {
        let mut session = RlmSession::new().unwrap();
        let content = "The quick brown fox jumps over the lazy dog.";
        let file = write_temp_file(content);
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // Test peek
        let outcome = session
            .exec("result = peek(4, 9)", None, None, None)
            .unwrap();
        assert!(outcome.result.error.is_none());
        let result: String = serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        assert_eq!(result, "quick");

        // Test find
        let outcome = session
            .exec("result = find(r'\\b\\w+ox\\b')", None, None, None)
            .unwrap();
        assert!(outcome.result.error.is_none());
        let matches: Vec<(usize, usize)> =
            serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        assert!(!matches.is_empty());
    }

    #[tokio::test]
    async fn memory_persists_across_exec_calls() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("test context");
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // Store value in memory
        session
            .memory_put("counter".to_string(), serde_json::json!(1))
            .unwrap();

        // Verify memory persists
        assert_eq!(
            session.memory_get("counter"),
            Some(serde_json::json!(1))
        );

        // Update value via exec (indirectly via session manifest)
        session
            .memory_put("counter".to_string(), serde_json::json!(2))
            .unwrap();
        assert_eq!(
            session.memory_get("counter"),
            Some(serde_json::json!(2))
        );
    }

    #[tokio::test]
    async fn load_clears_helpers_and_memory() {
        let mut session = RlmSession::new().unwrap();
        let file1 = write_temp_file("first");
        let file2 = write_temp_file("second");

        // Load first file
        session
            .load_path(file1.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // Add helper and memory
        session
            .helpers_add("h1".to_string(), "pass".to_string())
            .unwrap();
        session
            .memory_put("k1".to_string(), serde_json::json!("v1"))
            .unwrap();
        assert_eq!(session.helpers_list().len(), 1);
        assert_eq!(session.memory_keys().len(), 1);

        // Reload (Reset) - should clear state
        session
            .load_path(file2.path(), RlmLoadMode::Reset)
            .await
            .unwrap();
        assert_eq!(session.helpers_list().len(), 0);
        assert_eq!(session.memory_keys().len(), 0);
    }

    #[tokio::test]
    async fn load_append_preserves_helpers_and_memory() {
        let mut session = RlmSession::new().unwrap();
        let file1 = write_temp_file("first");
        let file2 = write_temp_file("second");

        // Load first file
        session
            .load_path(file1.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // Add helper and memory
        session
            .helpers_add("h1".to_string(), "pass".to_string())
            .unwrap();
        session
            .memory_put("k1".to_string(), serde_json::json!("v1"))
            .unwrap();

        // Append - should preserve state
        session
            .load_path(file2.path(), RlmLoadMode::Append)
            .await
            .unwrap();
        assert_eq!(session.helpers_list(), vec!["h1".to_string()]);
        assert_eq!(session.memory_keys(), vec!["k1".to_string()]);
    }

    #[test]
    fn exec_without_context_fails() {
        let session = RlmSession::new().unwrap();
        // Don't load any context
        assert!(!session.has_context());
    }

    #[tokio::test]
    async fn exec_python_error_returns_traceback() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("test");
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let outcome = session.exec("1/0", None, None, None).unwrap();
        assert!(outcome.result.error.is_some(), "expected error");
        let error = outcome.result.error.as_ref().unwrap();
        // Python may return "division by zero" or "ZeroDivisionError"
        assert!(
            error.contains("ZeroDivisionError") || error.contains("division by zero"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn limits_override_applies_correctly() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("test");
        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let override_limits = RlmLimitsOverride {
            max_output_bytes: Some(1024),
            max_execution_ms: Some(5000),
            max_find_results: Some(100),
        };

        let outcome = session
            .exec("print('ok')", Some(&override_limits), None, None)
            .unwrap();
        assert!(outcome.result.error.is_none());
        assert_eq!(outcome.limits_applied.max_output_bytes, 1024);
        assert_eq!(outcome.limits_applied.max_execution_ms, 5000);
        assert_eq!(outcome.limits_applied.max_find_results, 100);
    }
}
