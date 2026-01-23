//! Cost tracking and analysis for RLM.
//!
//! This module provides comprehensive cost tracking for LLM API usage,
//! including input/output tokens, caching benefits, and budget enforcement.

use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

/// Cost per million tokens in micro-dollars.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct TokenPricing {
    /// Cost per 1M input tokens.
    pub input_per_m: u64,
    /// Cost per 1M output tokens.
    pub output_per_m: u64,
    /// Cost per 1M cached input tokens (if applicable).
    pub cached_input_per_m: Option<u64>,
}

impl TokenPricing {
    /// Create pricing with standard input/output rates.
    pub fn new(input_per_m: u64, output_per_m: u64) -> Self {
        Self {
            input_per_m,
            output_per_m,
            cached_input_per_m: None,
        }
    }

    /// Add cached input pricing.
    pub fn with_cached(mut self, cached_per_m: u64) -> Self {
        self.cached_input_per_m = Some(cached_per_m);
        self
    }

    /// Calculate cost for token usage.
    pub fn calculate(&self, input: u64, output: u64, cached_input: u64) -> u64 {
        let uncached_input = input.saturating_sub(cached_input);

        let input_cost = (uncached_input * self.input_per_m) / 1_000_000;
        let output_cost = (output * self.output_per_m) / 1_000_000;
        let cached_cost = if let Some(cached_rate) = self.cached_input_per_m {
            (cached_input * cached_rate) / 1_000_000
        } else {
            0
        };

        input_cost + output_cost + cached_cost
    }
}

/// Known model pricing (micro-dollars per 1M tokens).
pub fn default_pricing() -> HashMap<String, TokenPricing> {
    let mut pricing = HashMap::new();

    // Anthropic models
    pricing.insert(
        "claude-3-opus".to_string(),
        TokenPricing::new(15_000_000, 75_000_000).with_cached(1_500_000),
    );
    pricing.insert(
        "claude-3-sonnet".to_string(),
        TokenPricing::new(3_000_000, 15_000_000).with_cached(300_000),
    );
    pricing.insert(
        "claude-3-haiku".to_string(),
        TokenPricing::new(250_000, 1_250_000).with_cached(25_000),
    );
    pricing.insert(
        "claude-3-5-sonnet".to_string(),
        TokenPricing::new(3_000_000, 15_000_000).with_cached(300_000),
    );

    // OpenAI models
    pricing.insert(
        "gpt-4o".to_string(),
        TokenPricing::new(5_000_000, 15_000_000).with_cached(2_500_000),
    );
    pricing.insert(
        "gpt-4o-mini".to_string(),
        TokenPricing::new(150_000, 600_000).with_cached(75_000),
    );
    pricing.insert(
        "gpt-4-turbo".to_string(),
        TokenPricing::new(10_000_000, 30_000_000),
    );
    pricing.insert("o1".to_string(), TokenPricing::new(15_000_000, 60_000_000));
    pricing.insert(
        "o1-mini".to_string(),
        TokenPricing::new(3_000_000, 12_000_000),
    );

    pricing
}

/// Token usage for a single call.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input tokens (prompt).
    pub input_tokens: u64,
    /// Output tokens (completion).
    pub output_tokens: u64,
    /// Cached input tokens.
    pub cached_input_tokens: u64,
    /// Total tokens.
    pub total_tokens: u64,
}

impl TokenUsage {
    /// Create new token usage.
    pub fn new(input: u64, output: u64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            cached_input_tokens: 0,
            total_tokens: input + output,
        }
    }

    /// Set cached input tokens.
    pub fn with_cached(mut self, cached: u64) -> Self {
        self.cached_input_tokens = cached;
        self
    }
}

/// Cost breakdown for a single call.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Model used.
    pub model: String,
    /// Token usage.
    pub tokens: TokenUsage,
    /// Total cost in micro-dollars.
    pub cost_micros: u64,
    /// Cost saved by caching in micro-dollars.
    pub cache_savings_micros: u64,
}

/// Aggregated cost statistics.
#[derive(Debug, Default)]
pub struct CostStats {
    /// Total input tokens.
    input_tokens: AtomicU64,
    /// Total output tokens.
    output_tokens: AtomicU64,
    /// Total cached input tokens.
    cached_input_tokens: AtomicU64,
    /// Total cost in micro-dollars.
    total_cost_micros: AtomicU64,
    /// Total cache savings in micro-dollars.
    cache_savings_micros: AtomicU64,
    /// Number of API calls.
    api_calls: AtomicU64,
}

impl CostStats {
    /// Create new cost stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record token usage and cost.
    pub fn record(&self, tokens: &TokenUsage, cost: u64, cache_savings: u64) {
        self.input_tokens
            .fetch_add(tokens.input_tokens, Ordering::Relaxed);
        self.output_tokens
            .fetch_add(tokens.output_tokens, Ordering::Relaxed);
        self.cached_input_tokens
            .fetch_add(tokens.cached_input_tokens, Ordering::Relaxed);
        self.total_cost_micros.fetch_add(cost, Ordering::Relaxed);
        self.cache_savings_micros
            .fetch_add(cache_savings, Ordering::Relaxed);
        self.api_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Get a snapshot of current stats.
    ///
    /// Note: Uses Relaxed ordering for individual loads. The snapshot provides
    /// a consistent view where total_tokens equals input_tokens + output_tokens.
    pub fn snapshot(&self) -> CostStatsSnapshot {
        // Load all values once to ensure consistency within the snapshot
        let input_tokens = self.input_tokens.load(Ordering::Relaxed);
        let output_tokens = self.output_tokens.load(Ordering::Relaxed);
        let cached_input_tokens = self.cached_input_tokens.load(Ordering::Relaxed);
        let total_cost_micros = self.total_cost_micros.load(Ordering::Relaxed);
        let cache_savings_micros = self.cache_savings_micros.load(Ordering::Relaxed);
        let api_calls = self.api_calls.load(Ordering::Relaxed);

        CostStatsSnapshot {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            total_tokens: input_tokens + output_tokens,
            total_cost_micros,
            total_cost_dollars: total_cost_micros as f64 / 1_000_000.0,
            cache_savings_micros,
            cache_savings_dollars: cache_savings_micros as f64 / 1_000_000.0,
            api_calls,
        }
    }
}

/// Serializable snapshot of cost stats.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CostStatsSnapshot {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_input_tokens: u64,
    pub total_tokens: u64,
    pub total_cost_micros: u64,
    pub total_cost_dollars: f64,
    pub cache_savings_micros: u64,
    pub cache_savings_dollars: f64,
    pub api_calls: u64,
}

impl CostStatsSnapshot {
    /// Get cost per 1000 tokens.
    pub fn cost_per_1k_tokens(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }
        (self.total_cost_dollars / self.total_tokens as f64) * 1000.0
    }

    /// Get average cost per API call in dollars.
    pub fn avg_cost_per_call(&self) -> f64 {
        if self.api_calls == 0 {
            return 0.0;
        }
        self.total_cost_dollars / self.api_calls as f64
    }

    /// Get cache utilization rate.
    pub fn cache_utilization(&self) -> f64 {
        if self.input_tokens == 0 {
            return 0.0;
        }
        (self.cached_input_tokens as f64 / self.input_tokens as f64) * 100.0
    }
}

/// Budget configuration and enforcement.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostBudget {
    /// Maximum total cost in micro-dollars.
    pub max_cost_micros: u64,
    /// Maximum tokens per request.
    pub max_tokens_per_request: Option<u64>,
    /// Maximum total tokens.
    pub max_total_tokens: Option<u64>,
    /// Alert threshold (percentage of budget).
    pub alert_threshold_pct: u8,
}

impl Default for CostBudget {
    fn default() -> Self {
        Self {
            max_cost_micros: 10_000_000, // $10 default
            max_tokens_per_request: Some(100_000),
            max_total_tokens: None,
            alert_threshold_pct: 80,
        }
    }
}

/// Budget status.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetStatus {
    /// Under budget, all clear.
    Ok,
    /// Approaching budget limit.
    Warning,
    /// At or over budget limit.
    Exceeded,
}

/// Cost tracker with budget enforcement.
pub struct CostTracker {
    /// Pricing information by model.
    pricing: RwLock<HashMap<String, TokenPricing>>,
    /// Per-model statistics.
    model_stats: RwLock<HashMap<String, CostStats>>,
    /// Aggregate statistics.
    total_stats: CostStats,
    /// Budget configuration.
    budget: CostBudget,
}

impl CostTracker {
    /// Create a new cost tracker.
    pub fn new(budget: CostBudget) -> Self {
        Self {
            pricing: RwLock::new(default_pricing()),
            model_stats: RwLock::new(HashMap::new()),
            total_stats: CostStats::new(),
            budget,
        }
    }

    /// Add or update pricing for a model.
    pub fn set_pricing(&self, model: &str, pricing: TokenPricing) {
        if let Ok(mut p) = self.pricing.write() {
            p.insert(model.to_string(), pricing);
        }
    }

    /// Record a call and calculate cost.
    pub fn record_call(&self, model: &str, tokens: TokenUsage) -> CostBreakdown {
        let pricing = self
            .pricing
            .read()
            .ok()
            .and_then(|p| p.get(model).copied())
            .unwrap_or_default();

        // Calculate actual cost
        let cost = pricing.calculate(
            tokens.input_tokens,
            tokens.output_tokens,
            tokens.cached_input_tokens,
        );

        // Calculate what it would have cost without caching
        let uncached_cost = pricing.calculate(tokens.input_tokens, tokens.output_tokens, 0);

        let cache_savings = uncached_cost.saturating_sub(cost);

        // Record in total stats
        self.total_stats.record(&tokens, cost, cache_savings);

        // Record in per-model stats
        if let Ok(mut stats) = self.model_stats.write() {
            let model_stats = stats.entry(model.to_string()).or_default();
            model_stats.record(&tokens, cost, cache_savings);
        }

        CostBreakdown {
            model: model.to_string(),
            tokens,
            cost_micros: cost,
            cache_savings_micros: cache_savings,
        }
    }

    /// Check if a request would exceed budget.
    pub fn check_budget(&self, estimated_cost: u64) -> BudgetStatus {
        let current = self.total_stats.total_cost_micros.load(Ordering::Relaxed);
        let projected = current + estimated_cost;

        if projected >= self.budget.max_cost_micros {
            BudgetStatus::Exceeded
        } else if projected
            >= (self.budget.max_cost_micros * self.budget.alert_threshold_pct as u64) / 100
        {
            BudgetStatus::Warning
        } else {
            BudgetStatus::Ok
        }
    }

    /// Check if tokens would exceed per-request limit.
    pub fn check_token_limit(&self, tokens: u64) -> bool {
        if let Some(limit) = self.budget.max_tokens_per_request {
            tokens <= limit
        } else {
            true
        }
    }

    /// Get total statistics.
    pub fn total_stats(&self) -> CostStatsSnapshot {
        self.total_stats.snapshot()
    }

    /// Get per-model statistics.
    pub fn model_stats(&self) -> HashMap<String, CostStatsSnapshot> {
        self.model_stats
            .read()
            .map(|stats| {
                stats
                    .iter()
                    .map(|(k, v)| (k.clone(), v.snapshot()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get budget status.
    pub fn budget_status(&self) -> BudgetStatus {
        self.check_budget(0)
    }

    /// Get remaining budget in micro-dollars.
    pub fn remaining_budget(&self) -> u64 {
        let current = self.total_stats.total_cost_micros.load(Ordering::Relaxed);
        self.budget.max_cost_micros.saturating_sub(current)
    }

    /// Get remaining budget as percentage.
    pub fn remaining_budget_pct(&self) -> f64 {
        let remaining = self.remaining_budget();
        (remaining as f64 / self.budget.max_cost_micros as f64) * 100.0
    }
}

/// Estimate tokens from text (rough approximation).
pub fn estimate_tokens(text: &str) -> u64 {
    // Rough estimate: ~4 characters per token for English text
    (text.len() / 4).max(1) as u64
}

/// Generate a cost report.
pub fn generate_cost_report(tracker: &CostTracker) -> CostReport {
    let total = tracker.total_stats();
    let per_model = tracker.model_stats();

    CostReport {
        total,
        per_model,
        budget_remaining_pct: tracker.remaining_budget_pct(),
        budget_status: tracker.budget_status(),
    }
}

/// Cost report structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostReport {
    pub total: CostStatsSnapshot,
    pub per_model: HashMap<String, CostStatsSnapshot>,
    pub budget_remaining_pct: f64,
    pub budget_status: BudgetStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_pricing_calculation() {
        let pricing = TokenPricing::new(3_000_000, 15_000_000).with_cached(300_000);

        // 1000 input, 500 output, 0 cached
        // Input: 1000 * 3,000,000 / 1,000,000 = 3,000 micro-dollars
        // Output: 500 * 15,000,000 / 1,000,000 = 7,500 micro-dollars
        let cost = pricing.calculate(1000, 500, 0);
        assert_eq!(cost, 3_000 + 7_500);

        // 1000 input (800 cached), 500 output
        // Uncached: 200 * 3,000,000 / 1M = 600 micro
        // Output: 500 * 15,000,000 / 1M = 7,500 micro
        // Cached: 800 * 300,000 / 1M = 240 micro
        let cost_cached = pricing.calculate(1000, 500, 800);
        assert_eq!(cost_cached, 600 + 7_500 + 240);
        assert!(cost_cached < cost);
    }

    #[test]
    fn cost_tracker_records_calls() {
        let tracker = CostTracker::new(CostBudget::default());

        let tokens = TokenUsage::new(1000, 500).with_cached(800);
        let breakdown = tracker.record_call("claude-3-sonnet", tokens);

        assert!(!breakdown.model.is_empty());
        assert!(breakdown.cost_micros > 0);
        assert!(breakdown.cache_savings_micros > 0);

        let stats = tracker.total_stats();
        assert_eq!(stats.api_calls, 1);
        assert_eq!(stats.input_tokens, 1000);
        assert_eq!(stats.output_tokens, 500);
    }

    #[test]
    fn budget_checking() {
        let budget = CostBudget {
            max_cost_micros: 100,
            alert_threshold_pct: 80,
            ..Default::default()
        };
        let tracker = CostTracker::new(budget);

        // Start with OK
        assert_eq!(tracker.check_budget(0), BudgetStatus::Ok);

        // 79% should be OK
        assert_eq!(tracker.check_budget(79), BudgetStatus::Ok);

        // 80% should be Warning
        assert_eq!(tracker.check_budget(80), BudgetStatus::Warning);

        // 100% should be Exceeded
        assert_eq!(tracker.check_budget(100), BudgetStatus::Exceeded);
    }

    #[test]
    fn token_limit_checking() {
        let budget = CostBudget {
            max_tokens_per_request: Some(1000),
            ..Default::default()
        };
        let tracker = CostTracker::new(budget);

        assert!(tracker.check_token_limit(500));
        assert!(tracker.check_token_limit(1000));
        assert!(!tracker.check_token_limit(1001));
    }

    #[test]
    fn estimate_tokens_from_text() {
        assert!(estimate_tokens("Hello, world!") >= 1);
        let long_text = "a".repeat(400);
        assert_eq!(estimate_tokens(&long_text), 100);
    }

    #[test]
    fn cost_stats_snapshot() {
        let stats = CostStats::new();
        let tokens = TokenUsage::new(1000, 500);
        stats.record(&tokens, 100, 20);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.api_calls, 1);
        assert_eq!(snapshot.total_tokens, 1500);
        assert_eq!(snapshot.total_cost_micros, 100);
        assert!((snapshot.total_cost_dollars - 0.0001).abs() < 0.0001);
    }

    #[test]
    fn default_pricing_includes_common_models() {
        let pricing = default_pricing();

        assert!(pricing.contains_key("claude-3-opus"));
        assert!(pricing.contains_key("claude-3-sonnet"));
        assert!(pricing.contains_key("gpt-4o"));
        assert!(pricing.contains_key("gpt-4o-mini"));
    }

    #[test]
    fn per_model_stats() {
        let tracker = CostTracker::new(CostBudget::default());

        tracker.record_call("claude-3-sonnet", TokenUsage::new(100, 50));
        tracker.record_call("gpt-4o", TokenUsage::new(200, 100));
        tracker.record_call("claude-3-sonnet", TokenUsage::new(100, 50));

        let model_stats = tracker.model_stats();
        assert_eq!(model_stats.get("claude-3-sonnet").unwrap().api_calls, 2);
        assert_eq!(model_stats.get("gpt-4o").unwrap().api_calls, 1);
    }

    #[test]
    fn remaining_budget() {
        let budget = CostBudget {
            max_cost_micros: 1000,
            ..Default::default()
        };
        let tracker = CostTracker::new(budget);

        assert_eq!(tracker.remaining_budget(), 1000);

        // Manually record some cost
        tracker
            .total_stats
            .total_cost_micros
            .store(300, Ordering::Relaxed);
        assert_eq!(tracker.remaining_budget(), 700);
        assert!((tracker.remaining_budget_pct() - 70.0).abs() < 0.1);
    }
}
