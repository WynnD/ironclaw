use std::fmt;
use std::str::FromStr;

use chrono_tz::Tz;

use crate::config::helpers::{parse_bool_env, parse_optional_env, parse_string_env};
use crate::error::ConfigError;

/// Timezone used to interpret cron schedules for routines.
///
/// Routine timestamps remain stored in UTC; this only affects how cron expressions
/// (e.g. `0 0 9 * * *`) are interpreted.
#[derive(Debug, Clone, Default)]
pub enum RoutineCronTimezone {
    /// Use the host system's local timezone.
    #[default]
    Local,
    /// Interpret schedules in UTC.
    Utc,
    /// Interpret schedules in an explicit IANA timezone (e.g. America/New_York).
    Iana(Tz),
}

impl RoutineCronTimezone {
    pub fn parse(value: &str) -> Result<Self, ConfigError> {
        let normalized = value.trim();
        if normalized.is_empty() {
            return Ok(Self::Local);
        }

        let lower = normalized.to_ascii_lowercase();
        match lower.as_str() {
            "local" | "system" => Ok(Self::Local),
            "utc" | "gmt" | "z" => Ok(Self::Utc),
            _ => Tz::from_str(normalized)
                .map(Self::Iana)
                .map_err(|_| ConfigError::InvalidValue {
                    key: "ROUTINES_TIMEZONE".to_string(),
                    message: format!(
                        "must be 'local', 'utc', or an IANA timezone like 'America/Los_Angeles' (got '{normalized}')"
                    ),
                }),
        }
    }
}

impl fmt::Display for RoutineCronTimezone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoutineCronTimezone::Local => write!(f, "local"),
            RoutineCronTimezone::Utc => write!(f, "UTC"),
            RoutineCronTimezone::Iana(tz) => write!(f, "{tz}"),
        }
    }
}

/// Routines configuration.
#[derive(Debug, Clone)]
pub struct RoutineConfig {
    /// Whether the routines system is enabled.
    pub enabled: bool,
    /// How often (seconds) to poll for cron routines that need firing.
    pub cron_check_interval_secs: u64,
    /// Max routines executing concurrently across all users.
    pub max_concurrent_routines: usize,
    /// Timezone used to interpret cron schedules (stored timestamps remain UTC).
    pub cron_timezone: RoutineCronTimezone,
    /// Default cooldown between fires (seconds).
    pub default_cooldown_secs: u64,
    /// Max output tokens for lightweight routine LLM calls.
    pub max_lightweight_tokens: u32,
}

impl Default for RoutineConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cron_check_interval_secs: 15,
            max_concurrent_routines: 10,
            cron_timezone: RoutineCronTimezone::default(),
            default_cooldown_secs: 300,
            max_lightweight_tokens: 4096,
        }
    }
}

impl RoutineConfig {
    pub(crate) fn resolve() -> Result<Self, ConfigError> {
        let cron_timezone_raw = parse_string_env("ROUTINES_TIMEZONE", "local")?;
        Ok(Self {
            enabled: parse_bool_env("ROUTINES_ENABLED", true)?,
            cron_check_interval_secs: parse_optional_env("ROUTINES_CRON_INTERVAL", 15)?,
            max_concurrent_routines: parse_optional_env("ROUTINES_MAX_CONCURRENT", 10)?,
            cron_timezone: RoutineCronTimezone::parse(&cron_timezone_raw)?,
            default_cooldown_secs: parse_optional_env("ROUTINES_DEFAULT_COOLDOWN", 300)?,
            max_lightweight_tokens: parse_optional_env("ROUTINES_MAX_TOKENS", 4096)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::RoutineCronTimezone;

    #[test]
    fn test_parse_routine_timezone_local_aliases() {
        assert!(matches!(
            RoutineCronTimezone::parse("local").unwrap(),
            RoutineCronTimezone::Local
        ));
        assert!(matches!(
            RoutineCronTimezone::parse("system").unwrap(),
            RoutineCronTimezone::Local
        ));
    }

    #[test]
    fn test_parse_routine_timezone_utc_aliases() {
        assert!(matches!(
            RoutineCronTimezone::parse("utc").unwrap(),
            RoutineCronTimezone::Utc
        ));
        assert!(matches!(
            RoutineCronTimezone::parse("GMT").unwrap(),
            RoutineCronTimezone::Utc
        ));
    }

    #[test]
    fn test_parse_routine_timezone_iana() {
        let tz = RoutineCronTimezone::parse("America/Los_Angeles").unwrap();
        assert_eq!(tz.to_string(), "America/Los_Angeles");
    }
}
