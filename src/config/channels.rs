use std::collections::HashMap;
use std::path::PathBuf;

use secrecy::SecretString;

use crate::bootstrap::ironclaw_base_dir;
use crate::config::helpers::{optional_env, parse_bool_env, parse_optional_env};
use crate::error::ConfigError;
use crate::settings::Settings;

/// Channel configurations.
#[derive(Debug, Clone)]
pub struct ChannelsConfig {
    pub cli: CliConfig,
    pub http: Option<HttpConfig>,
    pub gateway: Option<GatewayConfig>,
    pub signal: Option<SignalConfig>,
    /// Directory containing WASM channel modules (default: ~/.ironclaw/channels/).
    pub wasm_channels_dir: std::path::PathBuf,
    /// Whether WASM channels are enabled.
    pub wasm_channels_enabled: bool,
    /// Per-channel owner user IDs. When set, the channel only responds to this user.
    /// Key: channel name (e.g., "telegram"), Value: owner user ID.
    pub wasm_channel_owner_ids: HashMap<String, i64>,
}

#[derive(Debug, Clone)]
pub struct CliConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct HttpConfig {
    pub host: String,
    pub port: u16,
    pub webhook_secret: Option<SecretString>,
    pub user_id: String,
}

/// Web gateway configuration.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    pub host: String,
    pub port: u16,
    /// Bearer token for authentication. Random hex generated at startup if unset.
    pub auth_token: Option<String>,
    pub user_id: String,
}

/// Signal channel configuration (signal-cli daemon HTTP/JSON-RPC).
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// Base URL of the signal-cli daemon HTTP endpoint (e.g. `http://127.0.0.1:8080`).
    pub http_url: String,
    /// Signal account identifier (E.164 phone number, e.g. `+1234567890`).
    pub account: String,
    /// Users allowed to interact with the bot in DMs.
    ///
    /// Each entry is one of:
    /// - `*` — allow everyone
    /// - E.164 phone number (e.g. `+1234567890`)
    /// - bare UUID (e.g. `a1b2c3d4-e5f6-7890-abcd-ef1234567890`)
    /// - `uuid:<id>` prefix form (e.g. `uuid:a1b2c3d4-e5f6-7890-abcd-ef1234567890`)
    ///
    /// An empty list denies all senders (secure by default).
    pub allow_from: Vec<String>,
    /// Groups allowed to interact with the bot.
    ///
    /// - Empty list — deny all group messages (DMs only, secure by default).
    /// - `*` — allow all groups.
    /// - Specific group IDs — allow only those groups.
    pub allow_from_groups: Vec<String>,
    /// DM policy: "open", "allowlist", or "pairing". Default: "pairing".
    ///
    /// - "open" — allow all DM senders (ignores allow_from for DMs)
    /// - "allowlist" — only allow senders in allow_from list
    /// - "pairing" — allowlist + send pairing reply to unknown users
    pub dm_policy: String,
    /// Group policy: "allowlist", "open", or "disabled". Default: "allowlist".
    ///
    /// - "disabled" — deny all group messages
    /// - "allowlist" — check allow_from_groups and group_allow_from
    /// - "open" — accept all group messages (respects allow_from_groups for group ID)
    pub group_policy: String,
    /// Allow list for group message senders. If empty, inherits from allow_from.
    pub group_allow_from: Vec<String>,
    /// Skip messages that contain only attachments (no text).
    pub ignore_attachments: bool,
    /// Skip story messages.
    pub ignore_stories: bool,
}

impl ChannelsConfig {
    pub(crate) fn resolve(settings: &Settings) -> Result<Self, ConfigError> {
        let http = if optional_env("HTTP_PORT")?.is_some() || optional_env("HTTP_HOST")?.is_some() {
            Some(HttpConfig {
                host: optional_env("HTTP_HOST")?.unwrap_or_else(|| "0.0.0.0".to_string()),
                port: parse_optional_env("HTTP_PORT", 8080)?,
                webhook_secret: optional_env("HTTP_WEBHOOK_SECRET")?.map(SecretString::from),
                user_id: optional_env("HTTP_USER_ID")?.unwrap_or_else(|| "http".to_string()),
            })
        } else {
            None
        };

        let gateway_enabled = parse_bool_env("GATEWAY_ENABLED", true)?;
        let gateway = if gateway_enabled {
            Some(GatewayConfig {
                host: optional_env("GATEWAY_HOST")?.unwrap_or_else(|| "127.0.0.1".to_string()),
                port: parse_optional_env("GATEWAY_PORT", 3000)?,
                auth_token: optional_env("GATEWAY_AUTH_TOKEN")?,
                user_id: optional_env("GATEWAY_USER_ID")?.unwrap_or_else(|| "default".to_string()),
            })
        } else {
            None
        };

        let signal = if let Some(http_url) = optional_env("SIGNAL_HTTP_URL")? {
            let account = optional_env("SIGNAL_ACCOUNT")?.ok_or(ConfigError::InvalidValue {
                key: "SIGNAL_ACCOUNT".to_string(),
                message: "SIGNAL_ACCOUNT is required when SIGNAL_HTTP_URL is set".to_string(),
            })?;
            let allow_from = match std::env::var_os("SIGNAL_ALLOW_FROM") {
                None => vec![account.clone()],
                Some(val) => {
                    let s = val.to_string_lossy();
                    s.split(',')
                        .map(|e| e.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                }
            };
            let dm_policy =
                optional_env("SIGNAL_DM_POLICY")?.unwrap_or_else(|| "pairing".to_string());
            let group_policy =
                optional_env("SIGNAL_GROUP_POLICY")?.unwrap_or_else(|| "allowlist".to_string());
            Some(SignalConfig {
                http_url,
                account,
                allow_from,
                allow_from_groups: optional_env("SIGNAL_ALLOW_FROM_GROUPS")?
                    .map(|s| {
                        s.split(',')
                            .map(|e| e.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect()
                    })
                    .unwrap_or_default(),
                dm_policy,
                group_policy,
                group_allow_from: optional_env("SIGNAL_GROUP_ALLOW_FROM")?
                    .map(|s| {
                        s.split(',')
                            .map(|e| e.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect()
                    })
                    .unwrap_or_default(),
                ignore_attachments: optional_env("SIGNAL_IGNORE_ATTACHMENTS")?
                    .map(|s| s.to_lowercase() == "true" || s == "1")
                    .unwrap_or(false),
                ignore_stories: optional_env("SIGNAL_IGNORE_STORIES")?
                    .map(|s| s.to_lowercase() == "true" || s == "1")
                    .unwrap_or(true),
            })
        } else {
            None
        };

        let cli_enabled = optional_env("CLI_ENABLED")?
            .map(|s| s.to_lowercase() != "false" && s != "0")
            .unwrap_or(true);

        Ok(Self {
            cli: CliConfig {
                enabled: cli_enabled,
            },
            http,
            gateway,
            signal,
            wasm_channels_dir: optional_env("WASM_CHANNELS_DIR")?
                .map(PathBuf::from)
                .unwrap_or_else(default_channels_dir),
            wasm_channels_enabled: parse_bool_env("WASM_CHANNELS_ENABLED", true)?,
            wasm_channel_owner_ids: {
                let mut ids = settings.channels.wasm_channel_owner_ids.clone();
                // Backwards compat: TELEGRAM_OWNER_ID env var
                if let Some(id_str) = optional_env("TELEGRAM_OWNER_ID")? {
                    let id: i64 = id_str.parse().map_err(|e: std::num::ParseIntError| {
                        ConfigError::InvalidValue {
                            key: "TELEGRAM_OWNER_ID".to_string(),
                            message: format!("must be an integer: {e}"),
                        }
                    })?;
                    ids.insert("telegram".to_string(), id);
                }
                ids
            },
        })
    }
}

/// Get the default channels directory (~/.ironclaw/channels/).
fn default_channels_dir() -> PathBuf {
    ironclaw_base_dir().join("channels")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::helpers::ENV_MUTEX;
    use crate::error::ConfigError;
    use crate::settings::Settings;

    fn clear_channel_env() {
        // SAFETY: Called only while holding ENV_MUTEX.
        unsafe {
            for key in [
                "HTTP_PORT",
                "HTTP_HOST",
                "HTTP_WEBHOOK_SECRET",
                "HTTP_USER_ID",
                "GATEWAY_ENABLED",
                "GATEWAY_HOST",
                "GATEWAY_PORT",
                "GATEWAY_AUTH_TOKEN",
                "GATEWAY_USER_ID",
                "SIGNAL_HTTP_URL",
                "SIGNAL_ACCOUNT",
                "SIGNAL_ALLOW_FROM",
                "SIGNAL_DM_POLICY",
                "SIGNAL_GROUP_POLICY",
                "SIGNAL_ALLOW_FROM_GROUPS",
                "SIGNAL_GROUP_ALLOW_FROM",
                "SIGNAL_IGNORE_ATTACHMENTS",
                "SIGNAL_IGNORE_STORIES",
                "CLI_ENABLED",
                "WASM_CHANNELS_DIR",
                "WASM_CHANNELS_ENABLED",
                "TELEGRAM_OWNER_ID",
            ] {
                std::env::remove_var(key);
            }
        }
    }

    #[test]
    fn gateway_user_id_uses_env_override() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_channel_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("GATEWAY_USER_ID", "gateway-abc");
        }

        let cfg = ChannelsConfig::resolve(&Settings::default()).expect("resolve should succeed");
        let gateway = cfg.gateway.expect("gateway should be enabled by default");
        assert_eq!(gateway.user_id, "gateway-abc");
    }

    #[test]
    fn signal_requires_account_when_http_url_set() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_channel_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("SIGNAL_HTTP_URL", "http://127.0.0.1:8080");
        }

        let err = ChannelsConfig::resolve(&Settings::default()).unwrap_err();
        assert!(matches!(
            err,
            ConfigError::InvalidValue { ref key, .. } if key == "SIGNAL_ACCOUNT"
        ));
    }

    #[test]
    fn signal_allow_from_defaults_to_account_when_unset() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_channel_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("SIGNAL_HTTP_URL", "http://127.0.0.1:8080");
            std::env::set_var("SIGNAL_ACCOUNT", "+15551230000");
        }

        let cfg = ChannelsConfig::resolve(&Settings::default()).expect("resolve should succeed");
        let signal = cfg.signal.expect("signal config should be present");
        assert_eq!(signal.account, "+15551230000");
        assert_eq!(signal.allow_from, vec!["+15551230000".to_string()]);
    }

    #[test]
    fn telegram_owner_id_invalid_value_errors() {
        let _guard = ENV_MUTEX.lock().expect("env mutex poisoned");
        clear_channel_env();
        // SAFETY: Under ENV_MUTEX.
        unsafe {
            std::env::set_var("TELEGRAM_OWNER_ID", "not-a-number");
        }

        let err = ChannelsConfig::resolve(&Settings::default()).unwrap_err();
        assert!(matches!(
            err,
            ConfigError::InvalidValue { ref key, .. } if key == "TELEGRAM_OWNER_ID"
        ));
    }
}
