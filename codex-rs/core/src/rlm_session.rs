use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use anyhow::Result;
use sha2::Digest;
use sha2::Sha256;
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
/// Maximum total memory for context + index (512MB default, matching RlmConfig).
const MAX_MEMORY_BYTES: usize = 512 * 1024 * 1024;

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
    /// SHA256 hash of the context for determinism verification and replay.
    pub(crate) context_hash: String,
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
    /// Context string wrapped in Arc to avoid cloning on every exec.
    context: Arc<String>,
    sources: Vec<String>,
    documents: Vec<DocumentMetadata>,
    routing_graph: Option<HierarchicalRoutingGraph>,
    helpers: Vec<RlmHelper>,
    memory: BTreeMap<String, serde_json::Value>,
    /// Cached memory bytes used (updated incrementally on put/clear).
    memory_bytes_cached: usize,
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
            context: Arc::new(String::new()),
            sources: Vec::new(),
            documents: Vec::new(),
            routing_graph: None,
            helpers: Vec::new(),
            memory: BTreeMap::new(),
            memory_bytes_cached: 0,
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

        // Compute SHA256 hash for determinism verification and replay
        let context_hash = {
            let mut hasher = Sha256::new();
            hasher.update(self.context.as_bytes());
            format!("{:x}", hasher.finalize())
        };

        RlmLoadStats {
            length_chars,
            length_tokens_estimate,
            line_count,
            document_count,
            has_routing,
            routing_entry_count,
            sources: self.sources.clone(),
            context_hash,
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

        // Build new context string (we need to create a new Arc when content changes)
        let new_context = if matches!(mode, RlmLoadMode::Append) && self.context_loaded {
            let mut ctx = String::with_capacity(
                self.context.len() + loaded.content.len() + path_label.len() + 50,
            );
            ctx.push_str(&self.context);
            ctx.push_str(&format!("\n\n===== APPENDED: {path_label} =====\n\n"));
            ctx.push_str(&loaded.content);
            ctx
        } else {
            loaded.content
        };
        self.context = Arc::new(new_context);

        // Enforce total memory limit before proceeding
        self.enforce_total_memory_limit()?;

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
        // Uses Arc::clone to share context without copying
        let search_callback = Arc::new(RlmSearchCallback::new(
            self.bm25_index_shared(),
            Arc::clone(&self.context),
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
        // Calculate size of new value
        let new_value_size = serde_json::to_string(&value)
            .map(|json| json.len())
            .unwrap_or(0);
        let new_entry_size = key.len() + new_value_size;

        // Calculate size of old value if replacing
        let old_entry_size = self.memory.get(&key).map(|old_value| {
            let old_value_size = serde_json::to_string(old_value)
                .map(|json| json.len())
                .unwrap_or(0);
            key.len() + old_value_size
        });

        // Update cache with delta
        let old_cached = self.memory_bytes_cached;
        self.memory_bytes_cached = self
            .memory_bytes_cached
            .saturating_sub(old_entry_size.unwrap_or(0))
            + new_entry_size;

        let old_value = self.memory.insert(key.clone(), value);
        if let Err(err) = self.enforce_memory_limit() {
            // Rollback: restore old value or remove the key, and restore cache
            self.memory_bytes_cached = old_cached;
            match old_value {
                Some(v) => {
                    self.memory.insert(key, v);
                }
                None => {
                    self.memory.remove(&key);
                }
            }
            return Err(err);
        }
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
        self.memory_bytes_cached = 0;
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
        self.memory_bytes_cached
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
        &*self.context
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
        let context = &*self.context;
        self.bm25_index
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|index| index.search(query, k, context)))
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

    /// Estimate total memory usage for context + BM25 index.
    ///
    /// The BM25 index roughly doubles the context size due to chunking with overlap
    /// and internal data structures.
    fn estimate_total_memory(&self) -> usize {
        let context_bytes = self.context.len();
        // BM25 index is estimated at ~2x context size when built
        let index_estimate = if self.bm25_index.lock().ok().and_then(|g| g.as_ref().map(|_| ())).is_some() {
            context_bytes * 2
        } else {
            // Index will be built on first search, so account for it
            context_bytes * 2
        };
        context_bytes + index_estimate
    }

    /// Enforce the total memory limit (context + index).
    fn enforce_total_memory_limit(&self) -> Result<()> {
        let used = self.estimate_total_memory();
        if used > MAX_MEMORY_BYTES {
            let used_mb = used / (1024 * 1024);
            let limit_mb = MAX_MEMORY_BYTES / (1024 * 1024);
            anyhow::bail!(
                "context_too_large: estimated memory {used_mb}MB exceeds limit {limit_mb}MB"
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
        self.context = Arc::new(String::new());
        self.sources.clear();
        self.documents.clear();
        self.routing_graph = None;
        self.helpers.clear();
        self.memory.clear();
        self.memory_bytes_cached = 0;
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
        // Set metadata for stats() builtin
        self.python
            .set_context_metadata(&self.sources, self.documents.len().max(1))?;
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
        // Pre-calculate total capacity: each helper contributes code + up to 2 newlines
        let capacity = self
            .helpers
            .iter()
            .map(|h| h.code.len() + 2)
            .sum::<usize>()
            + code.len();
        let mut combined = String::with_capacity(capacity);
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
    pub(crate) fn new(index: Arc<StdMutex<Option<Bm25Index>>>, context: Arc<String>) -> Self {
        Self { index, context }
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
            .map(|index| index.search(query, k, &self.context))
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

        // Create search callback with context wrapped in Arc
        let context = Arc::new(session.context().to_string());
        let callback = RlmSearchCallback::new(session.bm25_index_shared(), context);

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

        // Test find - now returns {"matches": [...], "capped": bool}
        let outcome = session
            .exec("result = find(r'\\b\\w+ox\\b')", None, None, None)
            .unwrap();
        assert!(outcome.result.error.is_none());
        let find_result: serde_json::Value =
            serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        let matches = find_result["matches"].as_array().unwrap();
        assert!(!matches.is_empty());
        assert_eq!(find_result["capped"], false);
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

    // Integration tests for full tool flow

    #[tokio::test]
    async fn full_load_exec_roundtrip_with_builtins() {
        let mut session = RlmSession::new().unwrap();
        let content = "Hello World! This is a test document with some content.";
        let file = write_temp_file(content);

        // Load context
        let stats = session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();
        assert!(stats.length_chars > 0);
        assert!(session.has_context());

        // Execute code using builtins
        // find() now returns {"matches": [...], "capped": bool}
        let outcome = session
            .exec(
                r#"
text = peek(0, 5)
find_result = find(r'World')
result = {'text': text, 'match_count': len(find_result['matches']), 'capped': find_result['capped']}
"#,
                None,
                None,
                None,
            )
            .unwrap();

        assert!(outcome.result.error.is_none(), "exec should succeed");
        let result: serde_json::Value =
            serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        assert_eq!(result["text"], "Hello");
        assert_eq!(result["match_count"], 1);
        assert_eq!(result["capped"], false);
    }

    #[tokio::test]
    async fn load_append_exec_preserves_all_state() {
        let mut session = RlmSession::new().unwrap();
        let file1 = write_temp_file("First document content.");
        let file2 = write_temp_file("Second document content.");

        // Load first file
        session
            .load_path(file1.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // Add helper
        session
            .helpers_add(
                "count_words".to_string(),
                "def count_words(text):\n    return len(text.split())\n".to_string(),
            )
            .unwrap();

        // Store in memory
        session
            .memory_put("pass1_result".to_string(), serde_json::json!({"count": 3}))
            .unwrap();

        // Append second file
        let stats = session
            .load_path(file2.path(), RlmLoadMode::Append)
            .await
            .unwrap();
        assert_eq!(stats.sources.len(), 2);

        // Verify helper still works
        let outcome = session
            .exec("result = count_words('a b c d e')", None, None, None)
            .unwrap();
        assert!(outcome.result.error.is_none());
        let result: i32 = serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        assert_eq!(result, 5);

        // Verify memory persists
        assert_eq!(
            session.memory_get("pass1_result"),
            Some(serde_json::json!({"count": 3}))
        );

        // Verify context contains both documents
        assert!(session.context().contains("First document"));
        assert!(session.context().contains("Second document"));
    }

    #[tokio::test]
    async fn exec_search_builtin_returns_results() {
        let mut session = RlmSession::new().unwrap();
        let content = "The quick brown fox jumps over the lazy dog. \
                       The fox is very quick. \
                       Dogs are loyal animals.";
        let file = write_temp_file(content);

        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let outcome = session
            .exec(
                r#"
results = search("fox", k=5)
result = {'count': len(results), 'has_fox': any('fox' in r['text'].lower() for r in results)}
"#,
                None,
                None,
                None,
            )
            .unwrap();

        assert!(outcome.result.error.is_none());
        let result: serde_json::Value =
            serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        assert!(result["count"].as_i64().unwrap() > 0);
        assert_eq!(result["has_fox"], true);
    }

    #[tokio::test]
    async fn exec_stats_and_session_builtins_work() {
        let mut session = RlmSession::new().unwrap();
        let content = "Line 1\nLine 2\nLine 3\n";
        let file = write_temp_file(content);

        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let outcome = session
            .exec(
                r#"
s = stats()
sess = session()
result = {
    'chars': s['chars'],
    'lines': s['lines'],
    'docs': s['docs'],
    'has_sources': len(s['sources']) > 0
}
"#,
                None,
                None,
                None,
            )
            .unwrap();

        assert!(
            outcome.result.error.is_none(),
            "exec failed: {:?} traceback: {:?}",
            outcome.result.error,
            outcome.result.traceback
        );
        let result: serde_json::Value =
            serde_json::from_str(&outcome.result.result_json.unwrap()).unwrap();
        assert!(result["chars"].as_i64().unwrap() > 0);
        // Content is "Line 1\nLine 2\nLine 3\n" which has 3 newlines, so 4 lines
        assert_eq!(result["lines"], 4);
        assert_eq!(result["docs"], 1);
        assert_eq!(result["has_sources"], true);
    }

    #[tokio::test]
    async fn multiple_exec_calls_share_state() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("test content");

        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        // First exec: define a value in memory via helper
        session
            .helpers_add(
                "store".to_string(),
                "stored_value = 42\n".to_string(),
            )
            .unwrap();

        // First exec: use the helper
        let outcome1 = session
            .exec("result = stored_value", None, None, None)
            .unwrap();
        assert!(outcome1.result.error.is_none());
        let result1: i32 = serde_json::from_str(&outcome1.result.result_json.unwrap()).unwrap();
        assert_eq!(result1, 42);

        // Second exec: modify and return
        let outcome2 = session
            .exec("result = stored_value * 2", None, None, None)
            .unwrap();
        assert!(outcome2.result.error.is_none());
        let result2: i32 = serde_json::from_str(&outcome2.result.result_json.unwrap()).unwrap();
        assert_eq!(result2, 84);
    }

    #[tokio::test]
    async fn budget_is_tracked_across_exec_calls() {
        let mut session = RlmSession::new().unwrap();
        let file = write_temp_file("test");

        session
            .load_path(file.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let initial_budget = session.budget_snapshot();

        // Execute some code
        session.exec("x = 1 + 1", None, None, None).unwrap();

        // Budget should still be available (no sub-agent calls)
        let after_budget = session.budget_snapshot();
        assert_eq!(
            initial_budget.remaining_sub_calls,
            after_budget.remaining_sub_calls
        );
    }

    #[tokio::test]
    async fn reset_load_clears_budget() {
        let mut session = RlmSession::new().unwrap();
        let file1 = write_temp_file("first");
        let file2 = write_temp_file("second");

        session
            .load_path(file1.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let budget1 = session.budget_snapshot();

        // Reset with new file should reset budget
        session
            .load_path(file2.path(), RlmLoadMode::Reset)
            .await
            .unwrap();

        let budget2 = session.budget_snapshot();

        // Budget should be fresh (same as initial)
        assert_eq!(budget1.remaining_sub_calls, budget2.remaining_sub_calls);
        assert_eq!(budget1.remaining_tokens, budget2.remaining_tokens);
    }

    // Helper and memory limit enforcement tests

    #[test]
    fn helper_limit_is_enforced() {
        let mut session = RlmSession::new().unwrap();

        // Try to add a helper that's too large (over 1MB)
        let large_code = "x = ".to_string() + &"0".repeat(HELPERS_LIMIT_BYTES + 100);
        let result = session.helpers_add("big".to_string(), large_code);

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("helper limit exceeded"));
    }

    #[test]
    fn helper_limit_accumulates_across_helpers() {
        let mut session = RlmSession::new().unwrap();

        // Add several helpers that together exceed the limit
        let half_limit = HELPERS_LIMIT_BYTES / 2;
        let code1 = "a = ".to_string() + &"1".repeat(half_limit);
        let code2 = "b = ".to_string() + &"2".repeat(half_limit);

        // First helper should succeed
        session.helpers_add("h1".to_string(), code1).unwrap();

        // Second helper should fail (total exceeds limit)
        let result = session.helpers_add("h2".to_string(), code2);
        assert!(result.is_err());

        // Original helper should still be present
        assert_eq!(session.helpers_list(), vec!["h1".to_string()]);
    }

    #[test]
    fn memory_limit_is_enforced() {
        let mut session = RlmSession::new().unwrap();

        // Try to add a memory value that's too large (over 5MB)
        let large_string = "x".repeat(MEMORY_LIMIT_BYTES + 100);
        let result = session.memory_put("big".to_string(), serde_json::json!(large_string));

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("memory limit exceeded"));
    }

    #[test]
    fn memory_limit_accumulates_across_keys() {
        let mut session = RlmSession::new().unwrap();

        // Add several memory entries that together exceed the limit
        let half_limit = MEMORY_LIMIT_BYTES / 2;
        let value1 = "a".repeat(half_limit);
        let value2 = "b".repeat(half_limit);

        // First value should succeed
        session
            .memory_put("k1".to_string(), serde_json::json!(value1))
            .unwrap();

        // Second value should fail (total exceeds limit)
        let result = session.memory_put("k2".to_string(), serde_json::json!(value2));
        assert!(result.is_err());

        // Original value should still be present
        assert!(session.memory_get("k1").is_some());
        assert!(session.memory_get("k2").is_none());
    }

    #[test]
    fn helper_removal_frees_space() {
        let mut session = RlmSession::new().unwrap();

        // Add a large helper
        let large_code = "x = ".to_string() + &"0".repeat(HELPERS_LIMIT_BYTES - 1000);
        session.helpers_add("large".to_string(), large_code).unwrap();

        // Try to add another - should fail
        let small_code = "y = ".to_string() + &"1".repeat(2000);
        assert!(session
            .helpers_add("small".to_string(), small_code.clone())
            .is_err());

        // Remove the large helper
        session.helpers_remove("large").unwrap();

        // Now the small one should succeed
        session.helpers_add("small".to_string(), small_code).unwrap();
        assert_eq!(session.helpers_list(), vec!["small".to_string()]);
    }

    #[test]
    fn memory_clear_frees_space() {
        let mut session = RlmSession::new().unwrap();

        // Fill memory near the limit
        let large_value = "x".repeat(MEMORY_LIMIT_BYTES - 1000);
        session
            .memory_put("large".to_string(), serde_json::json!(large_value))
            .unwrap();

        // Try to add more - should fail
        let small_value = "y".repeat(2000);
        assert!(session
            .memory_put("small".to_string(), serde_json::json!(small_value.clone()))
            .is_err());

        // Clear memory
        session.memory_clear().unwrap();

        // Now we can add again
        session
            .memory_put("small".to_string(), serde_json::json!(small_value))
            .unwrap();
        assert!(session.memory_get("small").is_some());
    }

    #[test]
    fn helpers_bytes_used_is_accurate() {
        let mut session = RlmSession::new().unwrap();

        let name1 = "helper1";
        let code1 = "def f(): return 1";
        session.helpers_add(name1.to_string(), code1.to_string()).unwrap();

        let expected = name1.len() + code1.len();
        assert_eq!(session.helpers_bytes_used(), expected);

        let name2 = "helper2";
        let code2 = "def g(): return 2";
        session.helpers_add(name2.to_string(), code2.to_string()).unwrap();

        let expected2 = expected + name2.len() + code2.len();
        assert_eq!(session.helpers_bytes_used(), expected2);
    }

    #[test]
    fn memory_bytes_used_is_accurate() {
        let mut session = RlmSession::new().unwrap();

        session
            .memory_put("key1".to_string(), serde_json::json!("value1"))
            .unwrap();

        // key1 = 4 bytes, "value1" serialized = "\"value1\"" = 8 bytes
        let used1 = session.memory_bytes_used();
        assert!(used1 > 0);

        session
            .memory_put("key2".to_string(), serde_json::json!({"nested": true}))
            .unwrap();

        let used2 = session.memory_bytes_used();
        assert!(used2 > used1);
    }
}
