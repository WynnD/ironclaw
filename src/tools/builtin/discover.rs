//! Discover tools for deferred tool loading.
//!
//! When deferred tool loading is enabled, only core tools are sent with every
//! LLM request. This tool allows the LLM to search for and activate additional
//! tools on demand, reducing payload size for providers with request limits.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;

use crate::context::JobContext;
use crate::tools::ToolRegistry;
use crate::tools::tool::{Tool, ToolError, ToolOutput};

/// Tool for discovering and activating deferred tools.
///
/// Returns matching tool names and descriptions. The agentic loop detects
/// calls to this tool and adds discovered tools to subsequent LLM requests.
pub struct DiscoverToolsTool {
    registry: Arc<ToolRegistry>,
}

impl DiscoverToolsTool {
    /// Create a new discover tools tool.
    pub fn new(registry: Arc<ToolRegistry>) -> Self {
        Self { registry }
    }
}

#[async_trait]
impl Tool for DiscoverToolsTool {
    fn name(&self) -> &str {
        "discover_tools"
    }

    fn description(&self) -> &str {
        "Search for and activate additional tools by keyword or exact name. \
         Use when you need a capability not available in your current tool set. \
         Discovered tools become available for subsequent calls in this conversation."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword to search tool names and descriptions (e.g. 'routine', 'skill', 'extension')"
                },
                "names": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Exact tool names to activate (e.g. ['routine_create', 'routine_list'])"
                }
            }
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        _ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = Instant::now();

        let query = params.get("query").and_then(|v| v.as_str());
        let names: Vec<&str> = params
            .get("names")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        if query.is_none() && names.is_empty() {
            return Err(ToolError::InvalidParameters(
                "Provide 'query' (keyword search) and/or 'names' (exact tool names)".into(),
            ));
        }

        let mut results: Vec<serde_json::Value> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Search by keyword
        if let Some(q) = query {
            let found = self.registry.search_tools(q).await;
            for def in found {
                if seen.insert(def.name.clone()) {
                    results.push(serde_json::json!({
                        "name": def.name,
                        "description": def.description,
                    }));
                }
            }
        }

        // Resolve exact names
        if !names.is_empty() {
            let name_refs: Vec<&str> = names.to_vec();
            let found = self.registry.tool_definitions_for(&name_refs).await;
            for def in found {
                if seen.insert(def.name.clone()) {
                    results.push(serde_json::json!({
                        "name": def.name,
                        "description": def.description,
                    }));
                }
            }
        }

        let output = if results.is_empty() {
            serde_json::json!({
                "found": 0,
                "tools": [],
                "hint": "No tools matched. Try broader keywords or check the deferred tools list in the system prompt."
            })
        } else {
            serde_json::json!({
                "found": results.len(),
                "tools": results,
                "hint": "These tools are now activated and available for use in this conversation."
            })
        };

        Ok(ToolOutput::success(output, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false
    }

    fn is_core(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_discover_tools_requires_params() {
        let registry = Arc::new(ToolRegistry::new());
        let tool = DiscoverToolsTool::new(registry);
        let ctx = JobContext::default();

        let result = tool.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_discover_tools_empty_query() {
        let registry = Arc::new(ToolRegistry::new());
        let tool = DiscoverToolsTool::new(registry);
        let ctx = JobContext::default();

        let result = tool
            .execute(serde_json::json!({"query": "nonexistent"}), &ctx)
            .await
            .unwrap();

        let found = result.result.get("found").unwrap().as_u64().unwrap();
        assert_eq!(found, 0);
    }
}
