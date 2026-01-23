//! RLM: Recursive Language Model Runtime
//!
//! This crate implements the embedded runtime used by Codex RLM tools: Python
//! execution helpers, context loading, routing metadata, and budget tracking.

pub mod config;
pub mod context;
pub mod cost;
pub mod error;
pub mod index;
pub mod policy;
pub mod python;
pub mod routing;

pub use config::BudgetConfig;
pub use config::GatewayConfig;
pub use config::PromptConfig;
pub use config::ResolvedBudgetConfig;
pub use config::RlmConfig;
pub use config::RoutingConfig;
pub use config::SafetyConfig;
pub use config::WireApi;
pub use policy::BudgetSnapshot;
pub use policy::Capability;
pub use policy::DeterminismPolicy;
pub use policy::PolicyDiff;
pub use policy::PolicyLimits;
pub use policy::PolicyManifest;
pub use policy::PolicySummary;
pub use policy::PolicyViolation;
pub use policy::PolicyViolationType;
pub use routing::RoutingEntry;
pub use routing::RoutingGraph;
pub use routing::RoutingKind;

// Error types
pub use error::BudgetExceededKind;
pub use error::RlmError;

// Index types
pub use index::Bm25Index;
pub use index::Chunk;
pub use index::IndexConfig;
pub use index::SearchResult;
pub use index::SearchResultJson;

// Python runtime
pub use python::LlmCallback;
pub use python::SearchCallback;

// Cost tracking
pub use cost::BudgetStatus;
pub use cost::CostBreakdown;
pub use cost::CostBudget;
pub use cost::CostReport;
pub use cost::CostStats;
pub use cost::CostStatsSnapshot;
pub use cost::CostTracker;
pub use cost::TokenPricing;
pub use cost::TokenUsage;
pub use cost::estimate_tokens;
pub use cost::generate_cost_report;
