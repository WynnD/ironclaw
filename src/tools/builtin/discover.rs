//! Discover tools for deferred tool loading.
//!
//! When deferred tool loading is enabled, only core tools are sent with every
//! LLM request. This tool allows the LLM to search for and activate additional
//! tools on demand, reducing payload size for providers with request limits.

use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde_json::{Map, Value};

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
                },
                "include_schema": {
                    "type": "boolean",
                    "description": "Include each discovered tool's full parameters schema in the result",
                    "default": true
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
        let include_schema = params
            .get("include_schema")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

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
                    let mut item = serde_json::json!({
                        "name": def.name,
                        "description": def.description,
                        "parameters_summary": summarize_parameters_schema(&def.parameters),
                    });
                    if include_schema && let Some(obj) = item.as_object_mut() {
                        obj.insert("parameters_schema".to_string(), def.parameters);
                    }
                    results.push(item);
                }
            }
        }

        // Resolve exact names
        if !names.is_empty() {
            let name_refs: Vec<&str> = names.to_vec();
            let found = self.registry.tool_definitions_for(&name_refs).await;
            for def in found {
                if seen.insert(def.name.clone()) {
                    let mut item = serde_json::json!({
                        "name": def.name,
                        "description": def.description,
                        "parameters_summary": summarize_parameters_schema(&def.parameters),
                    });
                    if include_schema && let Some(obj) = item.as_object_mut() {
                        obj.insert("parameters_schema".to_string(), def.parameters);
                    }
                    results.push(item);
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

fn summarize_parameters_schema(schema: &Value) -> Value {
    let mut out = Map::new();

    let required_values = schema
        .get("required")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut required_names: Vec<String> = required_values
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    required_names.sort();
    required_names.dedup();
    out.insert(
        "required".to_string(),
        Value::Array(
            required_names
                .iter()
                .map(|name| Value::String(name.clone()))
                .collect(),
        ),
    );

    if let Some(props) = schema.get("properties").and_then(|v| v.as_object()) {
        let mut properties = Map::new();
        for (name, prop_schema) in props {
            let ty = prop_schema
                .get("type")
                .cloned()
                .unwrap_or_else(|| Value::String("any".to_string()));
            let desc = prop_schema
                .get("description")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            let required = Value::Bool(required_names.iter().any(|n| n == name));
            let enum_values = prop_schema.get("enum").cloned().unwrap_or(Value::Null);

            let mut info = Map::new();
            info.insert("type".to_string(), ty);
            info.insert("required".to_string(), required);
            info.insert("description".to_string(), desc);
            if !enum_values.is_null() {
                info.insert("enum".to_string(), enum_values);
            }

            properties.insert(name.clone(), Value::Object(info));
        }
        out.insert("properties".to_string(), Value::Object(properties));
    } else {
        out.insert("properties".to_string(), Value::Object(Map::new()));
    }

    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::tool::ApprovalRequirement;

    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            "mock_tool"
        }

        fn description(&self) -> &str {
            "Mock tool for discover_tools tests"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "repo": { "type": "string", "description": "Repository name" },
                    "dry_run": { "type": "boolean", "description": "Run without applying" }
                },
                "required": ["repo"]
            })
        }

        async fn execute(
            &self,
            _params: serde_json::Value,
            _ctx: &JobContext,
        ) -> Result<ToolOutput, ToolError> {
            Ok(ToolOutput::text("ok", std::time::Duration::from_millis(1)))
        }

        fn requires_approval(&self, _params: &serde_json::Value) -> ApprovalRequirement {
            ApprovalRequirement::Never
        }
    }

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

    #[tokio::test]
    async fn test_discover_tools_includes_parameter_details() {
        let registry = Arc::new(ToolRegistry::new());
        registry.register_sync(Arc::new(MockTool));
        let tool = DiscoverToolsTool::new(registry);
        let ctx = JobContext::default();

        let result = tool
            .execute(serde_json::json!({"query": "mock"}), &ctx)
            .await
            .unwrap();

        let tools = result
            .result
            .get("tools")
            .and_then(|v| v.as_array())
            .expect("tools array");
        assert_eq!(tools.len(), 1);

        let first = tools[0].as_object().expect("tool object");
        assert!(first.contains_key("parameters_schema"));
        let summary = first
            .get("parameters_summary")
            .and_then(|v| v.as_object())
            .expect("parameters_summary");
        let required = summary
            .get("required")
            .and_then(|v| v.as_array())
            .expect("required array");
        assert!(required.iter().any(|v| v.as_str() == Some("repo")));
    }
}
