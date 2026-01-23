//! Error types for the RLM runtime.

use thiserror::Error;

/// Errors that can occur during RLM execution.
#[derive(Debug, Error)]
pub enum RlmError {
    /// Budget limit exceeded.
    #[error("budget exceeded: {kind}")]
    BudgetExceeded {
        /// Which budget limit was exceeded.
        kind: BudgetExceededKind,
        /// Current remaining value when exceeded.
        remaining: u64,
        /// Amount that was requested.
        requested: u64,
    },

    /// Python execution failed.
    #[error("python execution failed: {message}")]
    PythonExecution {
        /// Error message from Python.
        message: String,
        /// Python traceback if available.
        traceback: Option<String>,
    },

    /// Disallowed Python import attempted.
    #[error("disallowed import: {module}")]
    DisallowedImport {
        /// The module that was attempted to be imported.
        module: String,
    },

    /// Recursion depth limit exceeded.
    #[error("recursion depth exceeded: depth={depth}, max={max_depth}")]
    RecursionDepthExceeded {
        /// Current depth.
        depth: u32,
        /// Maximum allowed depth.
        max_depth: u32,
    },

    /// Gateway/model call failed.
    #[error("gateway error: {message}")]
    Gateway {
        /// Error message.
        message: String,
        /// Provider name.
        provider: Option<String>,
    },

    /// Context loading failed.
    #[error("context error: {message}")]
    Context {
        /// Error message.
        message: String,
    },

    /// Configuration error.
    #[error("configuration error: {message}")]
    Config {
        /// Error message.
        message: String,
    },

    /// Generic internal error.
    #[error("internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// Types of budget limits that can be exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetExceededKind {
    /// Token budget exceeded.
    Tokens,
    /// Sub-call count exceeded.
    SubCalls,
    /// Wall-clock time exceeded.
    Time,
}

impl std::fmt::Display for BudgetExceededKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokens => write!(f, "tokens"),
            Self::SubCalls => write!(f, "sub_calls"),
            Self::Time => write!(f, "time"),
        }
    }
}

/// Result type alias for RLM operations.
pub type Result<T> = std::result::Result<T, RlmError>;

/// Error category for recovery decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Transient error that may succeed on retry (network issues, rate limits).
    Transient,
    /// Permanent error that won't succeed on retry (auth, config, syntax).
    Permanent,
    /// Resource exhaustion (budget exceeded, memory, time).
    ResourceExhausted,
    /// Security violation (sandbox escape attempt, injection).
    SecurityViolation,
}

/// Recovery action recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Retry the operation (with backoff).
    Retry,
    /// Fail immediately, don't retry.
    Fail,
    /// Return a partial/degraded result.
    Degrade,
    /// Skip this operation and continue.
    Skip,
}

impl RlmError {
    /// Categorize this error for recovery decisions.
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::BudgetExceeded { .. } => ErrorCategory::ResourceExhausted,
            Self::PythonExecution { .. } => ErrorCategory::Permanent,
            Self::DisallowedImport { .. } => ErrorCategory::SecurityViolation,
            Self::RecursionDepthExceeded { .. } => ErrorCategory::ResourceExhausted,
            Self::Gateway { message, .. } => {
                // Classify gateway errors based on message content
                let msg_lower = message.to_lowercase();
                if msg_lower.contains("rate limit")
                    || msg_lower.contains("timeout")
                    || msg_lower.contains("connection")
                    || msg_lower.contains("503")
                    || msg_lower.contains("502")
                {
                    ErrorCategory::Transient
                } else if msg_lower.contains("401")
                    || msg_lower.contains("403")
                    || msg_lower.contains("auth")
                {
                    ErrorCategory::Permanent
                } else {
                    ErrorCategory::Transient // Default to transient for unknown gateway errors
                }
            }
            Self::Context { .. } => ErrorCategory::Permanent,
            Self::Config { .. } => ErrorCategory::Permanent,
            Self::Internal(_) => ErrorCategory::Transient,
        }
    }

    /// Get recommended recovery action for this error.
    pub fn recovery_action(&self) -> RecoveryAction {
        match self.category() {
            ErrorCategory::Transient => RecoveryAction::Retry,
            ErrorCategory::Permanent => RecoveryAction::Fail,
            ErrorCategory::ResourceExhausted => RecoveryAction::Degrade,
            ErrorCategory::SecurityViolation => RecoveryAction::Fail,
        }
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        self.recovery_action() == RecoveryAction::Retry
    }

    /// Check if this error allows graceful degradation.
    pub fn allows_degradation(&self) -> bool {
        self.recovery_action() == RecoveryAction::Degrade
    }
}

/// Error recovery context for tracking retry state.
#[derive(Debug, Clone, Default)]
pub struct RecoveryContext {
    /// Number of retry attempts made.
    pub attempts: u32,
    /// Maximum retry attempts allowed.
    pub max_attempts: u32,
    /// Errors encountered during recovery.
    pub errors: Vec<String>,
    /// Whether degraded mode is active.
    pub degraded: bool,
}

impl RecoveryContext {
    /// Create a new recovery context with default settings.
    pub fn new(max_attempts: u32) -> Self {
        Self {
            attempts: 0,
            max_attempts,
            errors: Vec::new(),
            degraded: false,
        }
    }

    /// Record a retry attempt.
    pub fn record_attempt(&mut self, error: &RlmError) {
        self.attempts += 1;
        self.errors.push(error.to_string());
    }

    /// Check if more retries are allowed.
    pub fn can_retry(&self) -> bool {
        self.attempts < self.max_attempts
    }

    /// Enter degraded mode.
    pub fn enter_degraded_mode(&mut self) {
        self.degraded = true;
    }

    /// Check if operating in degraded mode.
    pub fn is_degraded(&self) -> bool {
        self.degraded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_exceeded_is_resource_exhausted() {
        let err = RlmError::BudgetExceeded {
            kind: BudgetExceededKind::Tokens,
            remaining: 0,
            requested: 100,
        };
        assert_eq!(err.category(), ErrorCategory::ResourceExhausted);
        assert_eq!(err.recovery_action(), RecoveryAction::Degrade);
    }

    #[test]
    fn gateway_rate_limit_is_transient() {
        let err = RlmError::Gateway {
            message: "rate limit exceeded".to_string(),
            provider: Some("openai".to_string()),
        };
        assert_eq!(err.category(), ErrorCategory::Transient);
        assert!(err.is_retryable());
    }

    #[test]
    fn gateway_auth_error_is_permanent() {
        let err = RlmError::Gateway {
            message: "401 unauthorized".to_string(),
            provider: Some("openai".to_string()),
        };
        assert_eq!(err.category(), ErrorCategory::Permanent);
        assert!(!err.is_retryable());
    }

    #[test]
    fn disallowed_import_is_security_violation() {
        let err = RlmError::DisallowedImport {
            module: "os".to_string(),
        };
        assert_eq!(err.category(), ErrorCategory::SecurityViolation);
        assert_eq!(err.recovery_action(), RecoveryAction::Fail);
    }

    #[test]
    fn recovery_context_tracks_attempts() {
        let mut ctx = RecoveryContext::new(3);
        assert!(ctx.can_retry());

        let err = RlmError::Gateway {
            message: "timeout".to_string(),
            provider: None,
        };

        ctx.record_attempt(&err);
        ctx.record_attempt(&err);
        assert!(ctx.can_retry());

        ctx.record_attempt(&err);
        assert!(!ctx.can_retry());
        assert_eq!(ctx.errors.len(), 3);
    }
}
