use crate::error::ConfigError;

use super::INJECTED_VARS;

/// Crate-wide mutex for tests that mutate process environment variables.
///
/// The process environment is global state shared across all threads.
/// Per-module mutexes do NOT prevent races between modules running in
/// parallel.  Every `unsafe { set_var / remove_var }` call in tests
/// MUST hold this single lock.
#[cfg(test)]
pub(crate) static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn optional_env(key: &str) -> Result<Option<String>, ConfigError> {
    // Check real env vars first (always win over injected secrets)
    match std::env::var(key) {
        Ok(val) if val.is_empty() => {}
        Ok(val) => return Ok(Some(val)),
        Err(std::env::VarError::NotPresent) => {}
        Err(e) => {
            return Err(ConfigError::ParseError(format!(
                "failed to read {key}: {e}"
            )));
        }
    }

    // Fall back to thread-safe overlay (secrets injected from DB)
    if let Some(val) = INJECTED_VARS.get().and_then(|map| map.get(key)) {
        return Ok(Some(val.clone()));
    }

    Ok(None)
}

pub(crate) fn parse_optional_env<T>(key: &str, default: T) -> Result<T, ConfigError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    optional_env(key)?
        .map(|s| {
            s.parse().map_err(|e| ConfigError::InvalidValue {
                key: key.to_string(),
                message: format!("{e}"),
            })
        })
        .transpose()
        .map(|opt| opt.unwrap_or(default))
}

/// Parse a boolean from an env var with a default.
///
/// Accepts "true"/"1" as true, "false"/"0" as false.
pub(crate) fn parse_bool_env(key: &str, default: bool) -> Result<bool, ConfigError> {
    match optional_env(key)? {
        Some(s) => match s.to_lowercase().as_str() {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => Err(ConfigError::InvalidValue {
                key: key.to_string(),
                message: format!("must be 'true' or 'false', got '{s}'"),
            }),
        },
        None => Ok(default),
    }
}

/// Parse an env var into `Option<T>` — returns `None` when unset,
/// `Some(parsed)` when set to a valid value.
pub(crate) fn parse_option_env<T>(key: &str) -> Result<Option<T>, ConfigError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    optional_env(key)?
        .map(|s| {
            s.parse().map_err(|e| ConfigError::InvalidValue {
                key: key.to_string(),
                message: format!("{e}"),
            })
        })
        .transpose()
}

/// Parse a string from an env var with a default.
pub(crate) fn parse_string_env(
    key: &str,
    default: impl Into<String>,
) -> Result<String, ConfigError> {
    Ok(optional_env(key)?.unwrap_or_else(|| default.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::helpers::ENV_MUTEX;

    fn clear_keys() {
        // SAFETY: Called only while holding ENV_MUTEX.
        unsafe {
            std::env::remove_var("TEST_BOOL");
            std::env::remove_var("TEST_NUM");
            std::env::remove_var("TEST_STR");
        }
    }

    #[test]
    fn parse_bool_env_parses_and_uses_default() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_keys();

        assert!(!parse_bool_env("TEST_BOOL", false).unwrap());
        assert!(parse_bool_env("TEST_BOOL", true).unwrap());

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("TEST_BOOL", "TrUe");
        }
        assert!(parse_bool_env("TEST_BOOL", false).unwrap());

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("TEST_BOOL", "0");
        }
        assert!(!parse_bool_env("TEST_BOOL", true).unwrap());
    }

    #[test]
    fn parse_bool_env_invalid_value_returns_error() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_keys();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("TEST_BOOL", "maybe");
        }

        let err = parse_bool_env("TEST_BOOL", false).unwrap_err();
        assert!(matches!(err, ConfigError::InvalidValue { ref key, .. } if key == "TEST_BOOL"));
    }

    #[test]
    fn parse_optional_and_option_env_parse_values() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_keys();

        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("TEST_NUM", "42");
        }

        let parsed_u64: u64 = parse_optional_env("TEST_NUM", 7_u64).unwrap();
        assert_eq!(parsed_u64, 42);

        let parsed_opt: Option<u64> = parse_option_env("TEST_NUM").unwrap();
        assert_eq!(parsed_opt, Some(42));
    }

    #[test]
    fn parse_string_env_uses_default_when_unset() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_keys();

        let s = parse_string_env("TEST_STR", "fallback").unwrap();
        assert_eq!(s, "fallback");
    }
}
