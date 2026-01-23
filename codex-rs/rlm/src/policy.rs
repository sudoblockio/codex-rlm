use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;

use crate::config::SafetyConfig;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyManifest {
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub capabilities: Vec<Capability>,
    pub allowed_modules: Vec<String>,
    pub limits: PolicyLimits,
    pub deterministic: DeterminismPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicySummary {
    pub capabilities: Vec<Capability>,
    pub allowed_modules: Vec<String>,
    pub limits: PolicyLimits,
    pub deterministic: DeterminismPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub remaining_tokens: u64,
    pub remaining_sub_calls: u32,
    pub remaining_tool_calls: u32,
    pub remaining_ms: u64,
}

impl BudgetSnapshot {
    pub fn new(
        remaining_tokens: u64,
        remaining_sub_calls: u32,
        remaining_tool_calls: u32,
        remaining_ms: u64,
    ) -> Self {
        Self {
            remaining_tokens,
            remaining_sub_calls,
            remaining_tool_calls,
            remaining_ms,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyLimits {
    pub max_depth: u32,
    pub max_sub_calls: u32,
    pub max_tool_calls_per_turn: u32,
    pub max_total_tokens: u64,
    pub max_output_bytes: u64,
    pub max_memory_mb: u64,
    pub max_cpu_seconds: u32,
    #[serde(default = "default_parallel_sub_calls")]
    pub parallel_sub_calls: bool,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: u32,
}

impl PolicyLimits {
    pub fn from_config(config: &SafetyConfig) -> Self {
        Self {
            max_depth: config.max_depth,
            max_sub_calls: config.max_sub_calls,
            max_tool_calls_per_turn: config.max_tool_calls_per_turn,
            max_total_tokens: config.max_total_tokens,
            max_output_bytes: config.max_output_bytes,
            max_memory_mb: config.max_memory_mb,
            max_cpu_seconds: config.max_cpu_seconds,
            parallel_sub_calls: config.parallel_sub_calls,
            max_concurrency: config.max_concurrency,
        }
    }
}

impl PolicyManifest {
    pub fn from_config(config: &SafetyConfig, version: impl Into<String>) -> Self {
        Self {
            version: version.into(),
            created_at: Utc::now(),
            capabilities: capabilities_from_config(config),
            allowed_modules: config.allowed_modules.clone(),
            limits: PolicyLimits::from_config(config),
            deterministic: DeterminismPolicy::from_config(config),
        }
    }
}

impl PolicySummary {
    pub fn from_config(config: &SafetyConfig) -> Self {
        Self {
            capabilities: capabilities_from_config(config),
            allowed_modules: config.allowed_modules.clone(),
            limits: PolicyLimits::from_config(config),
            deterministic: DeterminismPolicy::from_config(config),
        }
    }
}

impl DeterminismPolicy {
    pub fn from_config(config: &SafetyConfig) -> Self {
        Self {
            allow_clock: config.allow_clock,
            allow_randomness: config.allow_randomness,
            rng_seed: config.rng_seed,
        }
    }
}

fn capabilities_from_config(config: &SafetyConfig) -> Vec<Capability> {
    let mut caps = vec![Capability::StdlibImports];
    if config.allow_clock {
        caps.push(Capability::ClockAccess);
    }
    if config.allow_randomness {
        caps.push(Capability::Randomness);
    }
    caps
}

const fn default_parallel_sub_calls() -> bool {
    false
}

const fn default_max_concurrency() -> u32 {
    4
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeterminismPolicy {
    pub allow_clock: bool,
    pub allow_randomness: bool,
    pub rng_seed: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Capability {
    StdlibImports,
    FilesystemRead,
    FilesystemWrite,
    NetworkAccess,
    ClockAccess,
    Randomness,
    Subprocess,
}

impl Capability {
    pub fn as_str(&self) -> &'static str {
        match self {
            Capability::StdlibImports => "stdlib",
            Capability::FilesystemRead => "fs_read",
            Capability::FilesystemWrite => "fs_write",
            Capability::NetworkAccess => "network",
            Capability::ClockAccess => "clock",
            Capability::Randomness => "random",
            Capability::Subprocess => "subprocess",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyDiff {
    pub added: Vec<Capability>,
    pub removed: Vec<Capability>,
    pub changed_limits: Vec<String>,
    pub old_version: String,
    pub new_version: String,
}

impl PolicyDiff {
    pub fn between(previous: &PolicyManifest, current: &PolicyManifest) -> Self {
        let mut added = current
            .capabilities
            .iter()
            .filter(|cap| !previous.capabilities.contains(cap))
            .cloned()
            .collect::<Vec<_>>();
        added.sort_by_key(Capability::as_str);

        let mut removed = previous
            .capabilities
            .iter()
            .filter(|cap| !current.capabilities.contains(cap))
            .cloned()
            .collect::<Vec<_>>();
        removed.sort_by_key(Capability::as_str);

        let mut changed_limits = Vec::new();
        let previous_limits = &previous.limits;
        let current_limits = &current.limits;

        if previous_limits.max_depth != current_limits.max_depth {
            changed_limits.push("max_depth".to_string());
        }
        if previous_limits.max_sub_calls != current_limits.max_sub_calls {
            changed_limits.push("max_sub_calls".to_string());
        }
        if previous_limits.max_tool_calls_per_turn != current_limits.max_tool_calls_per_turn {
            changed_limits.push("max_tool_calls_per_turn".to_string());
        }
        if previous_limits.max_total_tokens != current_limits.max_total_tokens {
            changed_limits.push("max_total_tokens".to_string());
        }
        if previous_limits.max_output_bytes != current_limits.max_output_bytes {
            changed_limits.push("max_output_bytes".to_string());
        }
        if previous_limits.max_memory_mb != current_limits.max_memory_mb {
            changed_limits.push("max_memory_mb".to_string());
        }
        if previous_limits.max_cpu_seconds != current_limits.max_cpu_seconds {
            changed_limits.push("max_cpu_seconds".to_string());
        }
        if previous_limits.parallel_sub_calls != current_limits.parallel_sub_calls {
            changed_limits.push("parallel_sub_calls".to_string());
        }
        if previous_limits.max_concurrency != current_limits.max_concurrency {
            changed_limits.push("max_concurrency".to_string());
        }

        Self {
            added,
            removed,
            changed_limits,
            old_version: previous.version.clone(),
            new_version: current.version.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PolicyViolation {
    pub timestamp: DateTime<Utc>,
    pub turn: u32,
    pub violation_type: PolicyViolationType,
    pub detail: String,
    pub code_snippet: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PolicyViolationType {
    DisallowedImport,
    DisallowedBuiltin,
    DisallowedAttribute,
    FilesystemAccess,
    NetworkAccess,
    SubprocessAttempt,
    ResourceLimitExceeded,
    BudgetExceeded,
    RecursionDepthExceeded,
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn policy_manifest_round_trip() {
        let config = SafetyConfig::default();
        let manifest = PolicyManifest::from_config(&config, "v0.2");
        let json = serde_json::to_string(&manifest).unwrap();
        let decoded: PolicyManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest, decoded);
    }

    #[test]
    fn policy_summary_matches_limits() {
        let config = SafetyConfig::default();
        let summary = PolicySummary::from_config(&config);
        let limits = PolicyLimits::from_config(&config);
        assert_eq!(summary.limits, limits);
    }

    #[test]
    fn policy_diff_tracks_changes() {
        let mut config = SafetyConfig::default();
        let previous = PolicyManifest::from_config(&config, "v0.1");

        config.allow_clock = true;
        config.max_total_tokens = 400_000;
        let current = PolicyManifest::from_config(&config, "v0.2");

        let diff = PolicyDiff::between(&previous, &current);
        assert_eq!(diff.added, vec![Capability::ClockAccess]);
        assert_eq!(diff.removed, Vec::<Capability>::new());
        assert_eq!(diff.changed_limits, vec!["max_total_tokens".to_string()]);
        assert_eq!(diff.old_version, "v0.1");
        assert_eq!(diff.new_version, "v0.2");
    }
}
