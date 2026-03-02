use std::time::Duration;

use crate::config::helpers::{parse_bool_env, parse_option_env, parse_optional_env};
use crate::error::ConfigError;
use crate::settings::Settings;

/// Agent behavior configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub name: String,
    pub max_parallel_jobs: usize,
    pub job_timeout: Duration,
    pub stuck_threshold: Duration,
    pub repair_check_interval: Duration,
    pub max_repair_attempts: u32,
    /// Whether to use planning before tool execution.
    pub use_planning: bool,
    /// Session idle timeout. Sessions inactive longer than this are pruned.
    pub session_idle_timeout: Duration,
    /// Allow chat to use filesystem/shell tools directly (bypass sandbox).
    pub allow_local_tools: bool,
    /// Maximum daily LLM spend in cents (e.g. 10000 = $100). None = unlimited.
    pub max_cost_per_day_cents: Option<u64>,
    /// Maximum LLM/tool actions per hour. None = unlimited.
    pub max_actions_per_hour: Option<u64>,
    /// Maximum tool-call iterations per agentic loop invocation. Default 50.
    pub max_tool_iterations: usize,
    /// When true, skip tool approval checks entirely. For benchmarks/CI.
    pub auto_approve_tools: bool,
    /// Override the detected model context limit (tokens). `None` = auto-detect.
    pub context_limit_override: Option<usize>,
    /// Tokens to reserve for system/skills/output before history accounting.
    pub context_output_reserve_tokens: usize,
    /// Maximum estimated tokens from a single tool result allowed into LLM context.
    pub max_tool_output_tokens: usize,
    /// When true, only core tools are sent with each LLM request.
    /// Non-core tools are discoverable via `discover_tools` on demand.
    /// Reduces payload size for providers with request size limits.
    pub deferred_tool_loading: bool,
}

impl AgentConfig {
    pub(crate) fn resolve(settings: &Settings) -> Result<Self, ConfigError> {
        Ok(Self {
            name: parse_optional_env("AGENT_NAME", settings.agent.name.clone())?,
            max_parallel_jobs: parse_optional_env(
                "AGENT_MAX_PARALLEL_JOBS",
                settings.agent.max_parallel_jobs as usize,
            )?,
            job_timeout: Duration::from_secs(parse_optional_env(
                "AGENT_JOB_TIMEOUT_SECS",
                settings.agent.job_timeout_secs,
            )?),
            stuck_threshold: Duration::from_secs(parse_optional_env(
                "AGENT_STUCK_THRESHOLD_SECS",
                settings.agent.stuck_threshold_secs,
            )?),
            repair_check_interval: Duration::from_secs(parse_optional_env(
                "SELF_REPAIR_CHECK_INTERVAL_SECS",
                settings.agent.repair_check_interval_secs,
            )?),
            max_repair_attempts: parse_optional_env(
                "SELF_REPAIR_MAX_ATTEMPTS",
                settings.agent.max_repair_attempts,
            )?,
            use_planning: parse_bool_env("AGENT_USE_PLANNING", settings.agent.use_planning)?,
            session_idle_timeout: Duration::from_secs(parse_optional_env(
                "SESSION_IDLE_TIMEOUT_SECS",
                settings.agent.session_idle_timeout_secs,
            )?),
            allow_local_tools: parse_bool_env("ALLOW_LOCAL_TOOLS", false)?,
            max_cost_per_day_cents: parse_option_env("MAX_COST_PER_DAY_CENTS")?,
            max_actions_per_hour: parse_option_env("MAX_ACTIONS_PER_HOUR")?,
            max_tool_iterations: parse_optional_env(
                "AGENT_MAX_TOOL_ITERATIONS",
                settings.agent.max_tool_iterations,
            )?,
            auto_approve_tools: parse_bool_env(
                "AGENT_AUTO_APPROVE_TOOLS",
                settings.agent.auto_approve_tools,
            )?,
            context_limit_override: parse_option_env("CONTEXT_LIMIT")?,
            context_output_reserve_tokens: parse_optional_env(
                "CONTEXT_OUTPUT_RESERVE_TOKENS",
                4096usize,
            )?,
            max_tool_output_tokens: parse_optional_env("MAX_TOOL_OUTPUT_TOKENS", 4096usize)?,
            deferred_tool_loading: parse_bool_env("DEFERRED_TOOL_LOADING", false)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::helpers::ENV_MUTEX;
    use crate::settings::Settings;

    fn clear_agent_env() {
        // SAFETY: Called only while holding ENV_MUTEX.
        unsafe {
            for key in [
                "AGENT_NAME",
                "AGENT_MAX_PARALLEL_JOBS",
                "AGENT_JOB_TIMEOUT_SECS",
                "AGENT_STUCK_THRESHOLD_SECS",
                "SELF_REPAIR_CHECK_INTERVAL_SECS",
                "SELF_REPAIR_MAX_ATTEMPTS",
                "AGENT_USE_PLANNING",
                "SESSION_IDLE_TIMEOUT_SECS",
                "ALLOW_LOCAL_TOOLS",
                "MAX_COST_PER_DAY_CENTS",
                "MAX_ACTIONS_PER_HOUR",
                "AGENT_MAX_TOOL_ITERATIONS",
                "AGENT_AUTO_APPROVE_TOOLS",
                "CONTEXT_LIMIT",
                "CONTEXT_OUTPUT_RESERVE_TOKENS",
                "MAX_TOOL_OUTPUT_TOKENS",
                "DEFERRED_TOOL_LOADING",
            ] {
                std::env::remove_var(key);
            }
        }
    }

    #[test]
    fn resolve_uses_settings_defaults_when_env_unset() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_agent_env();

        let settings = Settings::default();
        let cfg = AgentConfig::resolve(&settings).expect("resolve should succeed");

        assert_eq!(cfg.name, settings.agent.name);
        assert_eq!(
            cfg.max_parallel_jobs,
            settings.agent.max_parallel_jobs as usize
        );
        assert_eq!(
            cfg.max_tool_iterations,
            settings.agent.max_tool_iterations
        );
        assert_eq!(cfg.auto_approve_tools, settings.agent.auto_approve_tools);
    }

    #[test]
    fn resolve_honors_env_overrides_for_numeric_and_bool_fields() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_agent_env();

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("AGENT_MAX_TOOL_ITERATIONS", "77");
            std::env::set_var("AGENT_AUTO_APPROVE_TOOLS", "true");
            std::env::set_var("MAX_COST_PER_DAY_CENTS", "12345");
            std::env::set_var("MAX_ACTIONS_PER_HOUR", "99");
            std::env::set_var("CONTEXT_LIMIT", "65536");
        }

        let cfg = AgentConfig::resolve(&Settings::default()).expect("resolve should succeed");
        assert_eq!(cfg.max_tool_iterations, 77);
        assert!(cfg.auto_approve_tools);
        assert_eq!(cfg.max_cost_per_day_cents, Some(12345));
        assert_eq!(cfg.max_actions_per_hour, Some(99));
        assert_eq!(cfg.context_limit_override, Some(65536));
    }
}
