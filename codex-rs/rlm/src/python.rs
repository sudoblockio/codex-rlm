use std::ffi::CString;

use anyhow::Result;
use anyhow::anyhow;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Serialize;

use crate::policy::BudgetSnapshot;
use crate::policy::PolicySummary;
use crate::routing::HierarchicalRoutingGraph;
use crate::routing::RoutingGraph;

#[derive(Clone, Debug)]
pub struct ExecutionResult {
    pub output: String,
    pub error: Option<String>,
    pub traceback: Option<String>,
    pub truncated: bool,
    pub timed_out: bool,
    pub find_results_capped: bool,
    pub result_json: Option<String>,
    pub result_meta_json: Option<String>,
    pub result_json_error: Option<String>,
    pub tool_override_events_json: Option<String>,
    pub tool_override_denied: bool,
}

pub trait LlmCallback: Send + Sync {
    fn call(&self, prompt: &str, tools: Option<Vec<String>>) -> Result<String>;
    fn batch(&self, prompts: Vec<String>) -> Result<Vec<String>>;
    fn budget_snapshot(&self) -> Result<BudgetSnapshot>;
}

/// Callback trait for BM25 search operations from Python.
pub trait SearchCallback: Send + Sync {
    /// Search the context using BM25.
    fn search(&self, query: &str, k: usize) -> Result<Vec<crate::index::SearchResultJson>>;
}

#[pyclass]
struct LlmHandler {
    callback: std::sync::Arc<dyn LlmCallback>,
}

#[pymethods]
impl LlmHandler {
    fn call(&self, py: Python<'_>, prompt: String, tools: Option<Vec<String>>) -> PyResult<String> {
        py.detach(|| self.callback.call(&prompt, tools))
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn batch(&self, py: Python<'_>, prompts: Vec<String>) -> PyResult<Vec<String>> {
        py.detach(|| self.callback.batch(prompts))
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    fn budget_snapshot_json(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| {
            let snapshot = self.callback.budget_snapshot()?;
            serde_json::to_string(&snapshot).map_err(anyhow::Error::from)
        })
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pyclass]
struct SearchHandler {
    callback: std::sync::Arc<dyn SearchCallback>,
}

#[pymethods]
impl SearchHandler {
    fn search(&self, py: Python<'_>, query: String, k: usize) -> PyResult<String> {
        py.detach(|| {
            let results = self.callback.search(&query, k)?;
            serde_json::to_string(&results).map_err(anyhow::Error::from)
        })
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

/// Resource limits for Python execution.
#[derive(Clone, Debug, Default)]
pub struct ResourceLimits {
    /// Maximum output bytes (0 = unlimited).
    pub max_output_bytes: u64,
    /// Maximum execution time in seconds (0 = unlimited).
    pub max_cpu_seconds: u32,
    /// Maximum number of results for find() (0 = unlimited).
    pub max_find_results: u32,
}

pub struct PythonRuntime {
    locals: Py<PyDict>,
    llm_callback: Option<std::sync::Arc<dyn LlmCallback>>,
    search_callback: Option<std::sync::Arc<dyn SearchCallback>>,
    resource_limits: ResourceLimits,
}

impl PythonRuntime {
    pub fn new() -> Result<Self> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let state = PyDict::new(py);
            locals.set_item("_state", &state)?;
            locals.set_item("P", "")?;
            let code = CString::new(Self::helpers_code())?;
            py.run(code.as_c_str(), Some(&locals), Some(&locals))?;
            Ok(Self {
                locals: locals.unbind(),
                llm_callback: None,
                search_callback: None,
                resource_limits: ResourceLimits::default(),
            })
        })
    }

    /// Set resource limits for Python execution.
    pub fn set_resource_limits(&mut self, limits: ResourceLimits) -> Result<()> {
        self.resource_limits = limits.clone();
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            state.set_item("max_output_bytes", limits.max_output_bytes)?;
            state.set_item("max_cpu_seconds", limits.max_cpu_seconds)?;
            state.set_item("max_find_results", limits.max_find_results)?;
            Ok(())
        })
    }

    /// Set the list of allowed Python modules for import.
    /// This installs a custom import finder that blocks disallowed imports.
    pub fn set_allowed_modules(&mut self, modules: &[String]) -> Result<()> {
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            let modules_list: Vec<&str> = modules.iter().map(String::as_str).collect();
            state.set_item("allowed_modules", modules_list)?;

            // Install the import restriction finder
            let setup_code = CString::new(Self::import_restriction_code())?;
            py.run(setup_code.as_c_str(), Some(locals), Some(locals))?;
            Ok(())
        })
    }

    pub fn set_context(&mut self, context: &str) -> Result<()> {
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            locals.set_item("P", context)?;
            Ok(())
        })
    }

    pub fn set_policy(&mut self, policy: PolicySummary) -> Result<()> {
        self.set_json("policy_json", &policy)
    }

    pub fn set_budget(&mut self, budget: BudgetSnapshot) -> Result<()> {
        self.set_json("budget_json", &budget)
    }

    pub fn set_state_json(&mut self, key: &str, json: &str) -> Result<()> {
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            state.set_item(key, json)?;
            Ok(())
        })
    }

    pub fn set_routing(&mut self, routing: RoutingGraph, summary: Option<String>) -> Result<()> {
        self.set_json("routing_json", &routing)?;
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            if let Some(summary) = summary {
                state.set_item("routing_summary", summary)?;
            } else {
                let _ = state.del_item("routing_summary");
            }
            Ok(())
        })
    }

    /// Set the hierarchical routing graph for find_routes() builtin.
    pub fn set_hierarchical_routing(&mut self, routing: &HierarchicalRoutingGraph) -> Result<()> {
        let json = serde_json::to_string(routing)?;
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            state.set_item("hierarchical_routing_json", json)?;
            Ok(())
        })
    }

    /// Set the document list for list_docs() builtin.
    pub fn set_document_list(&mut self, docs: &[crate::context::DocumentMetadata]) -> Result<()> {
        let json = serde_json::to_string(docs)?;
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            state.set_item("document_list_json", json)?;
            Ok(())
        })
    }

    pub fn set_llm_callback(&mut self, callback: std::sync::Arc<dyn LlmCallback>) -> Result<()> {
        self.llm_callback = Some(callback.clone());
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            let handler = Py::new(py, LlmHandler { callback })?;
            state.set_item("llm_handler", handler)?;
            Ok(())
        })
    }

    pub fn set_search_callback(&mut self, callback: std::sync::Arc<dyn SearchCallback>) -> Result<()> {
        self.search_callback = Some(callback.clone());
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            let handler = Py::new(py, SearchHandler { callback })?;
            state.set_item("search_handler", handler)?;
            Ok(())
        })
    }

    pub fn execute(&self, code: &str) -> Result<ExecutionResult> {
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            locals.set_item("_code", code)?;
            let wrapper = CString::new(Self::capture_code())?;
            py.run(wrapper.as_c_str(), Some(locals), Some(locals))?;
            let output_item = locals
                .get_item("_output")?
                .ok_or_else(|| anyhow!("missing _output"))?;
            let output: String = output_item.extract()?;
            let error = locals
                .get_item("_error")?
                .and_then(|item| item.extract::<Option<String>>().ok())
                .flatten();
            let traceback = locals
                .get_item("_traceback")?
                .and_then(|item| item.extract::<Option<String>>().ok())
                .flatten();
            let truncated = locals
                .get_item("_truncated")?
                .and_then(|item| item.extract::<Option<bool>>().ok())
                .flatten()
                .unwrap_or(false);
            let timed_out = locals
                .get_item("_timed_out")?
                .and_then(|item| item.extract::<Option<bool>>().ok())
                .flatten()
                .unwrap_or(false);
            let find_results_capped = locals
                .get_item("_find_results_capped")?
                .and_then(|item| item.extract::<Option<bool>>().ok())
                .flatten()
                .unwrap_or(false);
            let result_json = locals
                .get_item("_result_json")?
                .and_then(|item| item.extract::<Option<String>>().ok())
                .flatten();
            let result_meta_json = locals
                .get_item("_result_meta_json")?
                .and_then(|item| item.extract::<Option<String>>().ok())
                .flatten();
            let result_json_error = locals
                .get_item("_result_json_error")?
                .and_then(|item| item.extract::<Option<String>>().ok())
                .flatten();
            let tool_override_events_json = locals
                .get_item("_tool_override_events_json")?
                .and_then(|item| item.extract::<Option<String>>().ok())
                .flatten();
            let tool_override_denied = locals
                .get_item("_tool_override_denied")?
                .and_then(|item| item.extract::<Option<bool>>().ok())
                .flatten()
                .unwrap_or(false);
            let _ = locals.del_item("_output");
            let _ = locals.del_item("_error");
            let _ = locals.del_item("_traceback");
            let _ = locals.del_item("_truncated");
            let _ = locals.del_item("_timed_out");
            let _ = locals.del_item("_find_results_capped");
            let _ = locals.del_item("_result_json");
            let _ = locals.del_item("_result_meta_json");
            let _ = locals.del_item("_result_json_error");
            let _ = locals.del_item("_tool_override_events_json");
            let _ = locals.del_item("_tool_override_denied");
            let _ = locals.del_item("_code");
            Ok(ExecutionResult {
                output,
                error,
                traceback,
                truncated,
                timed_out,
                find_results_capped,
                result_json,
                result_meta_json,
                result_json_error,
                tool_override_events_json,
                tool_override_denied,
            })
        })
    }

    pub fn policy(&self) -> Result<Option<PolicySummary>> {
        self.decode_json("policy_json")
    }

    pub fn budget(&self) -> Result<Option<BudgetSnapshot>> {
        self.decode_json("budget_json")
    }

    pub fn routing(&self) -> Result<Option<RoutingGraph>> {
        self.decode_json("routing_json")
    }

    pub fn routing_summary(&self) -> Result<Option<String>> {
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            let Some(item) = state.get_item("routing_summary")? else {
                return Ok(None);
            };
            let summary: String = item.extract()?;
            Ok(Some(summary))
        })
    }

    fn set_json<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        let json = serde_json::to_string(value)?;
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            state.set_item(key, json)?;
            Ok(())
        })
    }

    fn decode_json<T: for<'de> serde::Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        Python::attach(|py| {
            let locals = self.locals.bind(py);
            let state_item = locals
                .get_item("_state")?
                .ok_or_else(|| anyhow!("missing _state"))?;
            let state = state_item
                .cast::<PyDict>()
                .map_err(|err| anyhow!(err.to_string()))?;
            let Some(item) = state.get_item(key)? else {
                return Ok(None);
            };
            let value: String = item.extract()?;
            Ok(Some(serde_json::from_str(&value)?))
        })
    }

    fn helpers_code() -> &'static str {
        r#"import json
import re


def _parse_flags(flags):
    if not flags:
        return 0
    value = 0
    if 'i' in flags:
        value |= re.IGNORECASE
    if 'm' in flags:
        value |= re.MULTILINE
    if 's' in flags:
        value |= re.DOTALL
    return value


def peek(start, end):
    return P[start:end]


def find(pattern, flags=None):
    compiled = re.compile(pattern, _parse_flags(flags))
    limit = _state.get("max_find_results", 0) or 0
    results = []
    for match in compiled.finditer(P):
        results.append((match.start(), match.end()))
        if limit > 0 and len(results) >= limit:
            _state["find_results_capped"] = True
            results.append((-1, -1))
            break
    return results


def stats():
    return {
        "length_chars": len(P),
        "length_tokens": max(1, len(P) // 4),
        "line_count": P.count("\n") + 1,
    }


def _decode_json(key):
    data = _state.get(key)
    if not data:
        return None
    return json.loads(data)


def policy():
    return _decode_json("policy_json")


def budget():
    return _decode_json("budget_json")

def session():
    return _decode_json("session_json")

def limits():
    return _decode_json("limits_json")


def routing():
    return _decode_json("routing_json")


def routing_summary():
    return _state.get("routing_summary")


def _record_tool_override(tools):
    events = _state.get("tool_override_events")
    if events is None:
        events = []
    events.append({
        "requested_tools": tools,
        "granted_tools": [],
        "denied_tools": tools,
        "reason": "tool_override_not_supported",
    })
    _state["tool_override_events"] = events
    _state["tool_override_denied"] = True


class PolicyViolationError(RuntimeError):
    pass


def _tool_override_policy():
    data = _state.get("tool_override_policy_json")
    if not data:
        return {"allowed_tool_overrides": []}
    try:
        return json.loads(data)
    except Exception:
        return {"allowed_tool_overrides": []}


def _evaluate_tool_override(tools):
    policy = _tool_override_policy()
    allowed = set(policy.get("allowed_tool_overrides") or [])
    requested = list(tools)
    blocked = [tool for tool in requested if tool.startswith("rlm_")]
    granted = [tool for tool in requested if tool in allowed and tool not in blocked]
    denied = [tool for tool in requested if tool not in allowed or tool in blocked]
    reason = None
    if blocked:
        reason = "rlm_tools_not_allowed"
    elif denied:
        reason = "requested tools not in allowed_tool_overrides"
    event = {
        "requested_tools": requested,
        "granted_tools": granted,
        "denied_tools": denied,
        "reason": reason,
    }
    return event, denied


def llm_query(prompt, tools=None):
    if tools is not None:
        event, denied = _evaluate_tool_override(tools)
        events = _state.get("tool_override_events")
        if events is None:
            events = []
        events.append(event)
        _state["tool_override_events"] = events
        if denied:
            _state["tool_override_denied"] = True
            raise PolicyViolationError("tool override denied")
    handler = _state.get("llm_handler")
    if handler is None:
        raise RuntimeError("llm_query is not configured")
    response = handler.call(prompt, tools)
    _state["budget_json"] = handler.budget_snapshot_json()
    return response


def llm_query_batch(prompts):
    handler = _state.get("llm_handler")
    if handler is None:
        raise RuntimeError("llm_query_batch is not configured")
    responses = handler.batch(prompts)
    _state["budget_json"] = handler.budget_snapshot_json()
    return responses


def search(query, k=10):
    """BM25 search over the loaded context.

    Args:
        query: Search query string
        k: Maximum number of results (default 10)

    Returns:
        List of dicts with keys: text, score, start, end
    """
    handler = _state.get("search_handler")
    if handler is None:
        raise RuntimeError("search is not configured - context may not be loaded")
    results_json = handler.search(query, k)
    return json.loads(results_json)


def find_routes(topic, limit=10):
    """Find routing entries matching a topic.

    Args:
        topic: Search query string
        limit: Maximum number of results (default 10)

    Returns:
        List of dicts with keys: agents_path, label, path, description, score, depth
    """
    data = _state.get("hierarchical_routing_json")
    if not data:
        return []

    graph = json.loads(data)
    nodes = graph.get("nodes", {})
    topic_lower = topic.lower()
    topic_words = topic_lower.split()

    matches = []
    for node_path, node in nodes.items():
        depth = node.get("depth", 0)
        for entry in node.get("entries", []):
            label = entry.get("label", "").lower()
            desc = entry.get("description", "").lower()
            path = entry.get("path", "").lower()

            score = 0.0
            # Exact phrase matches
            if topic_lower in label:
                score += 10.0
            if topic_lower in desc:
                score += 5.0
            if topic_lower in path:
                score += 3.0

            # Word matches
            for word in topic_words:
                if len(word) < 2:
                    continue
                if word in label:
                    score += 2.0
                if word in desc:
                    score += 1.0
                if word in path:
                    score += 0.5

            if score > 0:
                matches.append({
                    "agents_path": node_path,
                    "label": entry.get("label", ""),
                    "path": entry.get("path", ""),
                    "description": entry.get("description", ""),
                    "score": score,
                    "depth": depth,
                })

    # Sort by score descending, then depth ascending
    matches.sort(key=lambda x: (-x["score"], x["depth"]))
    return matches[:limit]


def list_docs(prefix=""):
    """List documents, optionally filtered by path prefix.

    Args:
        prefix: Optional path prefix to filter documents (e.g., "docs/api/")

    Returns:
        List of dicts with keys: id, size
    """
    data = _state.get("document_list_json")
    if not data:
        return []

    docs = json.loads(data)
    if prefix:
        docs = [d for d in docs if d.get("id", "").startswith(prefix)]

    return docs


def agents_files():
    """List all AGENTS.md files in the routing hierarchy.

    Returns:
        List of AGENTS.md file paths
    """
    data = _state.get("hierarchical_routing_json")
    if not data:
        return []

    graph = json.loads(data)
    return list(graph.get("nodes", {}).keys())


def route_path(to_path):
    """Get the navigation path from root to a given AGENTS.md.

    Args:
        to_path: Target AGENTS.md path

    Returns:
        List of AGENTS.md paths from root to target
    """
    data = _state.get("hierarchical_routing_json")
    if not data:
        return []

    graph = json.loads(data)
    nodes = graph.get("nodes", {})

    if to_path not in nodes:
        return []

    path = []
    current = to_path
    while current:
        path.append(current)
        node = nodes.get(current)
        if not node:
            break
        current = node.get("parent")

    path.reverse()
    return path
"#
    }

    fn capture_code() -> &'static str {
        r#"import ast
import json
import io
import sys
import time
import traceback
import builtins
import threading

# Resource limits from state
_max_output_bytes = _state.get("max_output_bytes", 0)
_max_cpu_seconds = _state.get("max_cpu_seconds", 0)

# Custom StringIO that enforces output limits
class _LimitedStringIO(io.StringIO):
    def __init__(self, max_bytes=0):
        super().__init__()
        self.max_bytes = max_bytes
        self.bytes_written = 0
        self.truncated = False

    def write(self, s):
        if self.truncated:
            return 0
        if self.max_bytes > 0:
            s_bytes = len(s.encode('utf-8', errors='replace'))
            if self.bytes_written + s_bytes > self.max_bytes:
                # Truncate to fit
                remaining = self.max_bytes - self.bytes_written
                if remaining > 0:
                    # Estimate chars from bytes (rough)
                    s = s[:remaining]
                    super().write(s)
                    super().write("\n... [output truncated: exceeded limit]")
                self.truncated = True
                return 0
            self.bytes_written += s_bytes
        return super().write(s)

_buf = _LimitedStringIO(_max_output_bytes)
_old_out = sys.stdout
_old_err = sys.stderr
sys.stdout = _buf
sys.stderr = _buf
_error = None
_traceback = None
_timed_out = False
_truncated = False
_find_results_capped = False
_result_json = None
_result_meta_json = None
_result_json_error = None
_tool_override_events_json = None
_tool_override_denied = False

# AST-based security validation
class _SecurityValidator(ast.NodeVisitor):
    """Validates AST for dangerous patterns before execution."""

    # Dangerous dunder attributes that could be used for sandbox escape
    BLOCKED_ATTRS = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__globals__', '__code__', '__closure__', '__func__',
        '__self__', '__dict__', '__builtins__', '__loader__',
        '__spec__', '__cached__', '__file__', '__path__',
        '__qualname__', '__module__', '__annotations__',
        '__init_subclass__', '__set_name__', '__reduce__',
        '__reduce_ex__', '__getstate__', '__setstate__',
    }

    # Functions that can be used for dynamic attribute access
    DANGEROUS_FUNCS = {'getattr', 'setattr', 'delattr', 'vars', 'dir'}

    def __init__(self):
        self.violations = []

    def visit_Attribute(self, node):
        """Check for dangerous attribute access patterns."""
        if node.attr in self.BLOCKED_ATTRS:
            self.violations.append(
                f"blocked attribute access: '{node.attr}' (line {node.lineno})"
            )
        self.generic_visit(node)

    def visit_Subscript(self, node):
        """Check for string subscript access to dunder names."""
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            if node.slice.value.startswith('__') and node.slice.value.endswith('__'):
                self.violations.append(
                    f"blocked subscript access: '{node.slice.value}' (line {node.lineno})"
                )
        self.generic_visit(node)

    def visit_Call(self, node):
        """Check for dangerous function calls."""
        # Check direct calls to dangerous functions
        if isinstance(node.func, ast.Name) and node.func.id in self.DANGEROUS_FUNCS:
            # Check if any argument is a string constant containing blocked attrs
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in self.BLOCKED_ATTRS:
                        self.violations.append(
                            f"blocked dynamic attribute access via {node.func.id}(): "
                            f"'{arg.value}' (line {node.lineno})"
                        )
        self.generic_visit(node)

    def visit_Import(self, node):
        """Record import statements for additional checks."""
        # Import validation is handled by the import restrictor
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Record from imports for additional checks."""
        # Import validation is handled by the import restrictor
        self.generic_visit(node)

def _validate_code(code_str):
    """Validate code using AST analysis. Returns list of violations."""
    try:
        tree = ast.parse(code_str)
        validator = _SecurityValidator()
        validator.visit(tree)
        return validator.violations
    except SyntaxError as e:
        return [f"syntax error: {e}"]

# Validate code before execution
_violations = _validate_code(_code)
if _violations:
    _error = "Security validation failed: " + "; ".join(_violations)
    _output = _error  # Also write to output so it's visible in the response
else:
    # Create restricted builtins (block dangerous functions)
    _BLOCKED_BUILTINS = {'open', 'exec', 'eval', 'compile', '__import__', 'input', 'breakpoint'}
    _restricted_builtins = {k: v for k, v in builtins.__dict__.items() if k not in _BLOCKED_BUILTINS}
    # Block exit/quit in non-interactive mode
    _restricted_builtins.pop('exit', None)
    _restricted_builtins.pop('quit', None)
    # Also block getattr/setattr/delattr to prevent dynamic attribute bypass
    _restricted_builtins.pop('getattr', None)
    _restricted_builtins.pop('setattr', None)
    _restricted_builtins.pop('delattr', None)
    _restricted_builtins.pop('vars', None)

    _start_time = time.time()
    try:
        _exec_globals = dict(globals())
        _exec_globals['__builtins__'] = _restricted_builtins
        _exec_globals.update(locals())
        _state["find_results_capped"] = False
        _state["tool_override_events"] = []
        _state["tool_override_denied"] = False
        # Remove internal variables from exec scope
        for _k in list(_exec_globals.keys()):
            if _k.startswith('_') and _k not in ('__builtins__', '__name__', '__doc__'):
                del _exec_globals[_k]
        # Re-add the required helper functions
        _exec_globals['peek'] = peek
        _exec_globals['find'] = find
        _exec_globals['stats'] = stats
        _exec_globals['policy'] = policy
        _exec_globals['budget'] = budget
        _exec_globals['session'] = session
        _exec_globals['limits'] = limits
        _exec_globals['routing'] = routing
        _exec_globals['routing_summary'] = routing_summary
        _exec_globals['llm_query'] = llm_query
        _exec_globals['llm_query_batch'] = llm_query_batch
        _exec_globals['search'] = search
        _exec_globals['find_routes'] = find_routes
        _exec_globals['list_docs'] = list_docs
        _exec_globals['agents_files'] = agents_files
        _exec_globals['route_path'] = route_path
        _exec_globals['P'] = P
        exec(_code, _exec_globals)
        if 'result' in _exec_globals:
            try:
                _result_json = json.dumps(_exec_globals.get('result'))
            except Exception as exc:
                _result_json_error = f"result: {exc}"
        if 'result_meta' in _exec_globals:
            try:
                _result_meta_json = json.dumps(_exec_globals.get('result_meta'))
            except Exception as exc:
                if _result_json_error:
                    _result_json_error = f"{_result_json_error}; result_meta: {exc}"
                else:
                    _result_json_error = f"result_meta: {exc}"
    except Exception as exc:
        _error = str(exc)
        _traceback = traceback.format_exc()
        traceback.print_exc()
    finally:
        sys.stdout = _old_out
        sys.stderr = _old_err

    _elapsed = time.time() - _start_time
    _output = _buf.getvalue()
    _truncated = _buf.truncated
    _find_results_capped = _state.get("find_results_capped", False)
    try:
        _tool_override_events_json = json.dumps(_state.get("tool_override_events", []))
    except Exception:
        _tool_override_events_json = "[]"
    _tool_override_denied = bool(_state.get("tool_override_denied", False))
    _state["tool_override_events"] = []
    _state["tool_override_denied"] = False

    # Check if execution exceeded time limit
    if _max_cpu_seconds > 0 and _elapsed > _max_cpu_seconds:
        _timed_out = True
        _output = f"[execution exceeded time limit: {_elapsed:.1f}s > {_max_cpu_seconds}s]\n" + _output
        if not _error:
            _error = f"execution timed out after {_elapsed:.1f}s"
"#
    }

    fn import_restriction_code() -> &'static str {
        r#"import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

class _RlmImportRestrictor(MetaPathFinder):
    """Custom import finder that blocks dangerous module imports.

    This restrictor takes a blocklist approach: it blocks explicitly dangerous
    modules (os, subprocess, etc.) while allowing everything else. This is more
    practical than an allowlist approach which would need to enumerate every
    internal Python module.
    """

    def __init__(self, allowed_modules):
        self.allowed_modules = set(allowed_modules)
        # Record modules already loaded before restriction
        self._preloaded = set(sys.modules.keys())
        # Dangerous modules that should always be blocked
        self._blocked = {
            # File system access
            'os', 'shutil', 'pathlib', 'glob', 'fnmatch',
            # Process/subprocess execution
            'subprocess', 'multiprocessing', 'concurrent',
            # Network access
            'socket', 'http', 'urllib', 'ftplib', 'smtplib', 'poplib',
            'imaplib', 'telnetlib', 'ssl', 'asyncio',
            # Low-level/unsafe
            'ctypes', 'cffi',
            # Package management
            'pip', 'setuptools', 'distutils', 'ensurepip',
            # Persistence that could write to disk
            'pickle', 'shelve', 'dbm', 'sqlite3',
            # Memory-mapped files
            'mmap',
            # Temp files (disk access)
            'tempfile',
            # Process/resource control
            'signal', 'resource', 'fcntl', 'termios', 'pty', 'tty',
            # User/group info
            'grp', 'pwd', 'spwd', 'crypt',
        }

    def _is_blocked(self, fullname):
        """Check if a module is explicitly blocked."""
        root = fullname.split('.')[0]
        return root in self._blocked

    def _is_allowed(self, fullname):
        root = fullname.split('.')[0]
        # Block dangerous modules first (even if preloaded)
        if self._is_blocked(fullname):
            return False
        # Allow preloaded modules and their submodules (stdlib loaded at startup)
        if root in self._preloaded or fullname in self._preloaded:
            return True
        # Allow modules starting with _ (internal Python modules)
        if fullname.startswith('_'):
            return True
        # Allow modules in the allowed list
        if root in self.allowed_modules or fullname in self.allowed_modules:
            return True
        # Allow all other modules by default (stdlib, etc.)
        # This is permissive but practical - true sandboxing needs a container
        return True

    def find_spec(self, fullname, path, target=None):
        if not self._is_allowed(fullname):
            raise ImportError(f"disallowed import: module '{fullname}' is blocked for security")
        # Return None to let other finders handle the import
        return None

# Install the restrictor if not already installed
_allowed = _state.get("allowed_modules", [])
if _allowed:
    # Remove any existing restrictor
    sys.meta_path = [f for f in sys.meta_path if not isinstance(f, _RlmImportRestrictor)]
    # Insert at the beginning to check first
    sys.meta_path.insert(0, _RlmImportRestrictor(_allowed))
"#
    }
}
