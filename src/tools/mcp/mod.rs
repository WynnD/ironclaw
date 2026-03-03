//! Model Context Protocol (MCP) integration.
//!
//! MCP allows the agent to connect to external tool servers that provide
//! additional capabilities through a standardized protocol.
//!
//! Supports both local (unauthenticated) and hosted (OAuth-authenticated) servers.
//!
//! ## Usage
//!
//! ```ignore
//! // Simple client (no auth)
//! let client = McpClient::new("http://localhost:8080");
//!
//! // Authenticated client (for hosted servers)
//! let client = McpClient::new_authenticated(
//!     config,
//!     session_manager,
//!     secrets,
//!     "user_id",
//! );
//!
//! // List and register tools
//! let tools = client.create_tools().await?;
//! for tool in tools {
//!     registry.register(tool);
//! }
//! ```

pub mod auth;
mod client;
pub mod config;
mod protocol;
pub mod session;

pub use auth::{is_authenticated, refresh_access_token};
pub use client::McpClient;
pub use config::{McpServerConfig, McpServersFile, OAuthConfig};
pub use protocol::{InitializeResult, McpRequest, McpResponse, McpTool};
pub use session::McpSessionManager;

/// Reserved MCP argument key for IronClaw-local execution hints.
pub const IRONCLAW_EXECUTION_HINT_KEY: &str = "__ironclaw_execution";

/// How IronClaw should run an MCP tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpExecutionMode {
    /// Execute normally and wait for the result.
    Blocking,
    /// Dispatch now and continue the chat loop without waiting.
    NonBlocking,
    /// Dispatch now and post the final result back into chat when complete.
    Background,
}

/// Parse execution hints from MCP tool parameters.
///
/// Supported shape:
/// `{ "__ironclaw_execution": { "mode": "blocking|non_blocking|background" } }`
pub fn mcp_execution_mode_from_params(params: &serde_json::Value) -> McpExecutionMode {
    let Some(hints) = params
        .as_object()
        .and_then(|obj| obj.get(IRONCLAW_EXECUTION_HINT_KEY))
        .and_then(serde_json::Value::as_object)
    else {
        return McpExecutionMode::Blocking;
    };

    match hints.get("mode").and_then(serde_json::Value::as_str) {
        Some("non_blocking") => McpExecutionMode::NonBlocking,
        Some("background") => McpExecutionMode::Background,
        _ => McpExecutionMode::Blocking,
    }
}

/// Remove IronClaw-local execution hints before forwarding arguments to MCP servers.
pub fn strip_mcp_execution_hints(params: &serde_json::Value) -> serde_json::Value {
    let Some(obj) = params.as_object() else {
        return params.clone();
    };

    let mut clean = obj.clone();
    clean.remove(IRONCLAW_EXECUTION_HINT_KEY);
    serde_json::Value::Object(clean)
}

#[cfg(test)]
mod tests {
    use super::{
        IRONCLAW_EXECUTION_HINT_KEY, McpExecutionMode, mcp_execution_mode_from_params,
        strip_mcp_execution_hints,
    };

    #[test]
    fn test_mcp_execution_mode_defaults_to_blocking() {
        let params = serde_json::json!({"query":"hello"});
        assert_eq!(
            mcp_execution_mode_from_params(&params),
            McpExecutionMode::Blocking
        );
    }

    #[test]
    fn test_mcp_execution_mode_non_blocking() {
        let params = serde_json::json!({
            IRONCLAW_EXECUTION_HINT_KEY: {
                "mode": "non_blocking"
            }
        });
        assert_eq!(
            mcp_execution_mode_from_params(&params),
            McpExecutionMode::NonBlocking
        );
    }

    #[test]
    fn test_mcp_execution_mode_background() {
        let params = serde_json::json!({
            "q": "status",
            IRONCLAW_EXECUTION_HINT_KEY: {
                "mode": "background"
            }
        });
        assert_eq!(
            mcp_execution_mode_from_params(&params),
            McpExecutionMode::Background
        );
    }

    #[test]
    fn test_strip_mcp_execution_hints() {
        let params = serde_json::json!({
            "query": "foo",
            IRONCLAW_EXECUTION_HINT_KEY: {
                "mode": "background"
            }
        });

        let stripped = strip_mcp_execution_hints(&params);
        assert_eq!(stripped, serde_json::json!({"query":"foo"}));
    }
}
