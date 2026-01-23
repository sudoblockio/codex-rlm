use crate::policy::Capability;
use crate::policy::PolicyLimits;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;

/// Main configuration for the RLM runtime.
///
/// This struct contains all configuration sections needed to run the RLM:
/// - `routing`: How to discover and load AGENTS.md manifests
/// - `safety`: Security limits and policy settings
/// - `budget`: Runtime budget constraints (tokens, time, etc.)
/// - `gateway`: Model provider settings (API keys, endpoints)
/// - `prompt`: System prompt template
#[derive(Clone, Debug, Default)]
pub struct RlmConfig {
    /// Routing configuration for AGENTS.md discovery.
    pub routing: RoutingConfig,
    /// Safety and policy configuration.
    pub safety: SafetyConfig,
    /// Runtime budget configuration.
    pub budget: BudgetConfig,
    /// Model gateway configuration.
    pub gateway: GatewayConfig,
    /// Prompt template configuration.
    pub prompt: PromptConfig,
}

impl RlmConfig {
    pub fn load_from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let file_config: RlmConfigFile = toml::from_str(&content)?;
        Ok(Self::from_file(file_config))
    }

    pub fn apply_codex_config_from_path(
        &mut self,
        path: &std::path::Path,
        apply_rlm_section: bool,
        prefer_codex_provider: bool,
    ) -> anyhow::Result<()> {
        let content = std::fs::read_to_string(path)?;
        let value: toml::Value = toml::from_str(&content)?;
        self.apply_codex_config_value(&value, apply_rlm_section, prefer_codex_provider)
    }

    fn from_file(file: RlmConfigFile) -> Self {
        let mut config = Self::default();
        config.apply_file(file);
        config
    }

    fn apply_file(&mut self, file: RlmConfigFile) {
        if let Some(routing) = file.routing {
            self.routing = routing.into();
        }
        if let Some(safety) = file.safety {
            self.safety = safety.into();
        }
        if let Some(budget) = file.budget {
            self.budget = budget.into();
        }
        if let Some(gateway) = file.gateway {
            self.gateway = gateway.into();
        }
        if let Some(prompt) = file.prompt {
            self.prompt = prompt.into();
        }
    }

    pub fn apply_codex_config_value(
        &mut self,
        value: &toml::Value,
        apply_rlm_section: bool,
        prefer_codex_provider: bool,
    ) -> anyhow::Result<()> {
        if apply_rlm_section && let Some(rlm_value) = value.get("rlm") {
            let file_config: RlmConfigFile = rlm_value.clone().try_into()?;
            self.apply_file(file_config);
        }
        self.apply_codex_model_providers(value, prefer_codex_provider)
    }

    fn apply_codex_model_providers(
        &mut self,
        value: &toml::Value,
        prefer_codex_provider: bool,
    ) -> anyhow::Result<()> {
        let mut providers: HashMap<String, CodexModelProviderInfo> = HashMap::new();
        if let Some(custom) = value.get("model_providers") {
            let custom_map: HashMap<String, CodexModelProviderInfo> = custom.clone().try_into()?;
            for (key, provider) in custom_map {
                providers.entry(key).or_insert(provider);
            }
        }

        if prefer_codex_provider
            && let Some(provider) = value.get("model_provider").and_then(toml::Value::as_str)
        {
            self.gateway.provider = provider.to_string();
        }

        if let Some(provider) = providers.get(&self.gateway.provider) {
            self.gateway.apply_codex_provider(provider);
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
struct CodexModelProviderInfo {
    pub base_url: Option<String>,
    pub env_key: Option<String>,
    pub experimental_bearer_token: Option<String>,
    pub wire_api: Option<WireApi>,
}

/// Configuration for routing manifest (AGENTS.md) discovery.
#[derive(Clone, Debug)]
pub struct RoutingConfig {
    /// Whether routing is enabled.
    pub enabled: bool,
    /// Whether to automatically discover manifests in the working directory.
    pub auto_load: bool,
    /// File names to look for when discovering manifests.
    pub manifest_paths: Vec<String>,
    /// Path to a specific manifest to use instead of auto-discovery.
    pub override_manifest: Option<String>,
    /// Glob patterns for prioritizing certain manifest locations.
    pub priority_globs: Vec<String>,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_load: true,
            manifest_paths: vec!["AGENTS.md".to_string()],
            override_manifest: None,
            priority_globs: vec![
                "docs/**".to_string(),
                "**/docs/AGENTS.md".to_string(),
                "**/docs/context.md".to_string(),
            ],
        }
    }
}

/// Safety and policy configuration for the RLM runtime.
///
/// These settings control what the Python sandbox can do and define
/// resource limits that are enforced during execution.
#[derive(Clone, Debug)]
pub struct SafetyConfig {
    /// Python modules allowed for import in the sandbox.
    pub allowed_modules: Vec<String>,
    /// Maximum recursion depth for sub-LM calls.
    pub max_depth: u32,
    /// Maximum number of sub-LM calls allowed.
    pub max_sub_calls: u32,
    /// Maximum tool calls per turn in the conversation loop.
    pub max_tool_calls_per_turn: u32,
    /// Maximum total tokens (input + output) across all calls.
    pub max_total_tokens: u64,
    /// Maximum output bytes from Python execution.
    pub max_output_bytes: u64,
    /// Maximum memory usage in megabytes.
    pub max_memory_mb: u64,
    /// Maximum CPU time in seconds.
    pub max_cpu_seconds: u32,
    /// Whether to allow parallel sub-LM calls (batch).
    pub parallel_sub_calls: bool,
    /// Maximum concurrent sub-LM calls when parallel is enabled.
    pub max_concurrency: u32,
    /// Whether to allow clock/time access in Python.
    pub allow_clock: bool,
    /// Whether to allow randomness in Python.
    pub allow_randomness: bool,
    /// Fixed seed for reproducible randomness.
    pub rng_seed: Option<u64>,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            allowed_modules: vec![
                "re".to_string(),
                "json".to_string(),
                "math".to_string(),
                "collections".to_string(),
                "itertools".to_string(),
                "functools".to_string(),
                "statistics".to_string(),
                "string".to_string(),
                "textwrap".to_string(),
                "difflib".to_string(),
                "heapq".to_string(),
                "bisect".to_string(),
                "typing".to_string(),
                "dataclasses".to_string(),
                "enum".to_string(),
            ],
            max_depth: 1,
            max_sub_calls: 50,
            max_tool_calls_per_turn: 20,
            max_total_tokens: 500_000,
            max_output_bytes: 10_000_000,
            max_memory_mb: 512,
            max_cpu_seconds: 30,
            parallel_sub_calls: false,
            max_concurrency: 4,
            allow_clock: false,
            allow_randomness: false,
            rng_seed: None,
        }
    }
}

impl SafetyConfig {
    pub fn capabilities_summary(&self) -> String {
        let mut caps = vec![Capability::StdlibImports];
        if self.allow_clock {
            caps.push(Capability::ClockAccess);
        }
        if self.allow_randomness {
            caps.push(Capability::Randomness);
        }
        caps.iter()
            .map(Capability::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn allowed_modules_summary(&self) -> String {
        self.allowed_modules.join(", ")
    }

    pub fn determinism_summary(&self) -> String {
        let mut flags = Vec::new();
        if self.allow_clock {
            flags.push("clock");
        }
        if self.allow_randomness {
            flags.push("random");
        }
        if flags.is_empty() {
            "strict".to_string()
        } else {
            flags.join(", ")
        }
    }

    pub fn limits_summary(&self) -> String {
        let limits = PolicyLimits::from_config(self);
        let max_depth = limits.max_depth;
        let max_sub_calls = limits.max_sub_calls;
        let max_tool_calls = limits.max_tool_calls_per_turn;
        let max_total_tokens = limits.max_total_tokens;
        let max_output_bytes = limits.max_output_bytes;
        let max_memory_mb = limits.max_memory_mb;
        let max_cpu_seconds = limits.max_cpu_seconds;
        let parallel_sub_calls = limits.parallel_sub_calls;
        let max_concurrency = limits.max_concurrency;
        format!(
            "depth={max_depth}, sub_calls={max_sub_calls}, tool_calls={max_tool_calls}, tokens={max_total_tokens}, output={max_output_bytes}, mem={max_memory_mb}MB, cpu={max_cpu_seconds}s, parallel_sub_calls={parallel_sub_calls}, max_concurrency={max_concurrency}",
        )
    }
}

/// Runtime budget configuration.
///
/// Note: `max_total_tokens`, `max_sub_calls`, and `max_tool_calls_per_turn` can be
/// set here to override the values from `SafetyConfig`. If not set (None), the
/// corresponding values from `SafetyConfig` will be used. This allows having
/// different limits for policy reporting vs. runtime enforcement.
#[derive(Clone, Debug, Default)]
pub struct BudgetConfig {
    /// Override for max_total_tokens (uses SafetyConfig value if None).
    pub max_total_tokens: Option<u64>,
    /// Override for max_sub_calls (uses SafetyConfig value if None).
    pub max_sub_calls: Option<u32>,
    /// Override for max_tool_calls_per_turn (uses SafetyConfig value if None).
    pub max_tool_calls_per_turn: Option<u32>,
    /// Maximum wall-clock time in milliseconds.
    pub max_ms: u64,
}

impl BudgetConfig {
    /// Create a resolved budget config using SafetyConfig as defaults.
    pub fn resolve(&self, safety: &SafetyConfig) -> ResolvedBudgetConfig {
        ResolvedBudgetConfig {
            max_total_tokens: self.max_total_tokens.unwrap_or(safety.max_total_tokens),
            max_sub_calls: self.max_sub_calls.unwrap_or(safety.max_sub_calls),
            max_tool_calls_per_turn: self
                .max_tool_calls_per_turn
                .unwrap_or(safety.max_tool_calls_per_turn),
            max_ms: if self.max_ms == 0 {
                300_000
            } else {
                self.max_ms
            },
        }
    }
}

/// Resolved budget configuration with concrete values.
#[derive(Clone, Debug)]
pub struct ResolvedBudgetConfig {
    pub max_total_tokens: u64,
    pub max_sub_calls: u32,
    pub max_tool_calls_per_turn: u32,
    pub max_ms: u64,
}

/// Configuration for the model gateway.
#[derive(Clone, Debug)]
pub struct GatewayConfig {
    /// Provider name (e.g., "openai", "anthropic").
    pub provider: String,
    /// Model name for root LM calls.
    pub model: String,
    /// Model name for sub-LM calls (recursive queries). Defaults to `model` if not set.
    pub sub_model: Option<String>,
    /// Base URL for the API.
    pub base_url: Option<String>,
    /// Environment variable name containing the API key.
    pub api_key_env: Option<String>,
    /// API key (use api_key_env instead for security).
    pub api_key: Option<String>,
    /// Wire API format to use.
    pub wire_api: WireApi,
    /// Maximum output tokens per response.
    pub max_output_tokens: u32,
}

impl GatewayConfig {
    /// Get the model to use for sub-calls.
    pub fn sub_model(&self) -> &str {
        self.sub_model.as_deref().unwrap_or(&self.model)
    }
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            sub_model: None, // Uses model by default
            base_url: None,
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            api_key: None,
            wire_api: WireApi::Responses,
            max_output_tokens: 1024,
        }
    }
}

impl GatewayConfig {
    fn apply_codex_provider(&mut self, provider: &CodexModelProviderInfo) {
        if self.base_url.is_none() {
            self.base_url = provider.base_url.clone();
        }
        if self.api_key_env.is_none() {
            self.api_key_env = provider.env_key.clone();
        }
        if self.api_key.is_none() {
            self.api_key = provider.experimental_bearer_token.clone();
        }
        if matches!(self.wire_api, WireApi::Responses)
            && let Some(wire_api) = provider.wire_api
        {
            self.wire_api = wire_api;
        }
    }
}

#[derive(Clone, Debug)]
pub struct PromptConfig {
    pub template: String,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            template: [
                "You are an RLM root model.",
                "Routing: {routing_summary}",
                "Policy: {policy_summary}",
                "Question: {query}",
            ]
            .join("\n"),
        }
    }
}

impl PromptConfig {
    pub fn render(&self, query: &str, routing_summary: &str, policy_summary: &str) -> String {
        self.template
            .replace("{query}", query)
            .replace("{routing_summary}", routing_summary)
            .replace("{policy_summary}", policy_summary)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WireApi {
    Responses,
    Chat,
    AnthropicMessages,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RlmConfigFile {
    pub routing: Option<RoutingConfigFile>,
    pub safety: Option<SafetyConfigFile>,
    pub budget: Option<BudgetConfigFile>,
    pub gateway: Option<GatewayConfigFile>,
    pub prompt: Option<PromptConfigFile>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RoutingConfigFile {
    pub enabled: Option<bool>,
    pub auto_load: Option<bool>,
    pub manifest_paths: Option<Vec<String>>,
    pub override_manifest: Option<String>,
    pub priority_globs: Option<Vec<String>>,
}

impl From<RoutingConfigFile> for RoutingConfig {
    fn from(file: RoutingConfigFile) -> Self {
        let mut config = RoutingConfig::default();
        if let Some(enabled) = file.enabled {
            config.enabled = enabled;
        }
        if let Some(auto_load) = file.auto_load {
            config.auto_load = auto_load;
        }
        if let Some(manifest_paths) = file.manifest_paths {
            config.manifest_paths = manifest_paths;
        }
        if let Some(override_manifest) = file.override_manifest {
            config.override_manifest = Some(override_manifest);
        }
        if let Some(priority_globs) = file.priority_globs {
            config.priority_globs = priority_globs;
        }
        config
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SafetyConfigFile {
    pub allowed_modules: Option<Vec<String>>,
    pub max_depth: Option<u32>,
    pub max_sub_calls: Option<u32>,
    pub max_tool_calls_per_turn: Option<u32>,
    pub max_total_tokens: Option<u64>,
    pub max_output_bytes: Option<u64>,
    pub max_memory_mb: Option<u64>,
    pub max_cpu_seconds: Option<u32>,
    pub parallel_sub_calls: Option<bool>,
    pub max_concurrency: Option<u32>,
    pub allow_clock: Option<bool>,
    pub allow_randomness: Option<bool>,
    pub rng_seed: Option<u64>,
}

impl From<SafetyConfigFile> for SafetyConfig {
    fn from(file: SafetyConfigFile) -> Self {
        let mut config = SafetyConfig::default();
        if let Some(allowed_modules) = file.allowed_modules {
            config.allowed_modules = allowed_modules;
        }
        if let Some(max_depth) = file.max_depth {
            config.max_depth = max_depth;
        }
        if let Some(max_sub_calls) = file.max_sub_calls {
            config.max_sub_calls = max_sub_calls;
        }
        if let Some(max_tool_calls_per_turn) = file.max_tool_calls_per_turn {
            config.max_tool_calls_per_turn = max_tool_calls_per_turn;
        }
        if let Some(max_total_tokens) = file.max_total_tokens {
            config.max_total_tokens = max_total_tokens;
        }
        if let Some(max_output_bytes) = file.max_output_bytes {
            config.max_output_bytes = max_output_bytes;
        }
        if let Some(max_memory_mb) = file.max_memory_mb {
            config.max_memory_mb = max_memory_mb;
        }
        if let Some(max_cpu_seconds) = file.max_cpu_seconds {
            config.max_cpu_seconds = max_cpu_seconds;
        }
        if let Some(parallel_sub_calls) = file.parallel_sub_calls {
            config.parallel_sub_calls = parallel_sub_calls;
        }
        if let Some(max_concurrency) = file.max_concurrency {
            config.max_concurrency = max_concurrency;
        }
        if let Some(allow_clock) = file.allow_clock {
            config.allow_clock = allow_clock;
        }
        if let Some(allow_randomness) = file.allow_randomness {
            config.allow_randomness = allow_randomness;
        }
        if let Some(rng_seed) = file.rng_seed {
            config.rng_seed = Some(rng_seed);
        }
        config
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BudgetConfigFile {
    pub max_total_tokens: Option<u64>,
    pub max_sub_calls: Option<u32>,
    pub max_tool_calls_per_turn: Option<u32>,
    pub max_ms: Option<u64>,
}

impl From<BudgetConfigFile> for BudgetConfig {
    fn from(file: BudgetConfigFile) -> Self {
        Self {
            max_total_tokens: file.max_total_tokens,
            max_sub_calls: file.max_sub_calls,
            max_tool_calls_per_turn: file.max_tool_calls_per_turn,
            max_ms: file.max_ms.unwrap_or(300_000),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GatewayConfigFile {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub sub_model: Option<String>,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub api_key: Option<String>,
    pub wire_api: Option<String>,
    pub max_output_tokens: Option<u32>,
}

impl From<GatewayConfigFile> for GatewayConfig {
    fn from(file: GatewayConfigFile) -> Self {
        let mut config = GatewayConfig::default();
        if let Some(provider) = file.provider {
            config.provider = provider;
        }
        if let Some(model) = file.model {
            config.model = model;
        }
        if let Some(sub_model) = file.sub_model {
            config.sub_model = Some(sub_model);
        }
        if let Some(base_url) = file.base_url {
            config.base_url = Some(base_url);
        }
        if let Some(api_key_env) = file.api_key_env {
            config.api_key_env = Some(api_key_env);
        }
        if let Some(api_key) = file.api_key {
            config.api_key = Some(api_key);
        }
        if let Some(wire_api) = file.wire_api {
            config.wire_api = wire_api.parse().unwrap_or(WireApi::Responses);
        }
        if let Some(max_output_tokens) = file.max_output_tokens {
            config.max_output_tokens = max_output_tokens;
        }
        config
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PromptConfigFile {
    pub template: Option<String>,
}

impl From<PromptConfigFile> for PromptConfig {
    fn from(file: PromptConfigFile) -> Self {
        let mut config = PromptConfig::default();
        if let Some(template) = file.template {
            config.template = template;
        }
        config
    }
}

impl std::str::FromStr for WireApi {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(match value {
            "chat" => Self::Chat,
            "anthropic" => Self::AnthropicMessages,
            _ => Self::Responses,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn renders_prompt_template() {
        let config = PromptConfig {
            template: "Q: {query} | R: {routing_summary} | P: {policy_summary}".to_string(),
        };
        let rendered = config.render("hello", "routing", "policy");
        assert_eq!(rendered, "Q: hello | R: routing | P: policy");
    }

    #[test]
    fn loads_gateway_config_from_file() {
        let toml = r#"
[gateway]
provider = "openai-compatible"
model = "gpt-test"
base_url = "http://localhost:11434/v1"
wire_api = "chat"
max_output_tokens = 64
"#;
        let file: RlmConfigFile = toml::from_str(toml).unwrap();
        let config = RlmConfig::from_file(file);
        assert_eq!(config.gateway.provider, "openai-compatible");
        assert_eq!(config.gateway.model, "gpt-test");
        assert_eq!(
            config.gateway.base_url.as_deref(),
            Some("http://localhost:11434/v1")
        );
        assert_eq!(config.gateway.wire_api, WireApi::Chat);
        assert_eq!(config.gateway.max_output_tokens, 64);
    }
}
