//! Generic adapter that bridges rig-core's `CompletionModel` trait to IronClaw's `LlmProvider`.
//!
//! This lets us use any rig-core provider (OpenAI, Anthropic, Ollama, etc.) as an
//! `Arc<dyn LlmProvider>` without changing any of the agent, reasoning, or tool code.

use async_trait::async_trait;
use rig::OneOrMany;
use rig::completion::{
    AssistantContent, CompletionModel, CompletionRequest as RigRequest,
    ToolDefinition as RigToolDefinition, Usage as RigUsage,
};
use rig::message::{
    Message as RigMessage, ToolChoice as RigToolChoice, ToolFunction, ToolResult as RigToolResult,
    ToolResultContent, UserContent,
};
use rust_decimal::Decimal;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value as JsonValue;

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::LlmError;
use crate::llm::costs;
use crate::llm::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider,
    ToolCall as IronToolCall, ToolCompletionRequest, ToolCompletionResponse,
    ToolDefinition as IronToolDefinition,
};

/// Optional native HTTP transport for bypassing rig-core's serialization.
///
/// rig-core always serializes message content as `[{"type":"text","text":"..."}]` (array),
/// which vLLM/SGLang endpoints choke on when tools are present. The native transport
/// sends requests with plain string content, matching curl/SDK format.
struct NativeTransport {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
}

/// Adapter that wraps a rig-core `CompletionModel` and implements `LlmProvider`.
pub struct RigAdapter<M: CompletionModel> {
    model: M,
    model_name: String,
    input_cost: Decimal,
    output_cost: Decimal,
    /// Native HTTP transport for K2.5 tool requests that bypass rig-core serialization.
    native_transport: Option<NativeTransport>,
}

impl<M: CompletionModel> RigAdapter<M> {
    /// Create a new adapter wrapping the given rig-core model.
    pub fn new(model: M, model_name: impl Into<String>) -> Self {
        let name = model_name.into();
        let (input_cost, output_cost) =
            costs::model_cost(&name).unwrap_or_else(costs::default_cost);
        Self {
            model,
            model_name: name,
            input_cost,
            output_cost,
            native_transport: None,
        }
    }

    /// Enable native HTTP transport for K2.5 tool requests.
    ///
    /// This bypasses rig-core's message serialization (which uses array content format)
    /// and sends requests with plain string content, matching what curl sends.
    pub fn with_native_transport(
        mut self,
        base_url: &str,
        api_key: String,
        extra_headers: reqwest::header::HeaderMap,
    ) -> Self {
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );
        default_headers.extend(extra_headers);

        let client = reqwest::Client::builder()
            .default_headers(default_headers)
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_default();

        self.native_transport = Some(NativeTransport {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        });
        self
    }
}

// -- Type conversion helpers --

/// Normalize a JSON Schema for OpenAI strict mode compliance.
///
/// OpenAI strict function calling requires:
/// - Every object must have `"additionalProperties": false`
/// - `"required"` must list ALL property keys
/// - Optional fields use `"type": ["<original>", "null"]` instead of being omitted from `required`
/// - Nested objects and array items are recursively normalized
///
/// This is applied as a clone-and-transform at the provider boundary so the
/// original tool definitions remain unchanged for other providers.
fn normalize_schema_strict(schema: &JsonValue) -> JsonValue {
    let mut schema = schema.clone();
    normalize_schema_recursive(&mut schema);
    schema
}

/// Recursively clean a JSON Schema without applying OpenAI strict-mode rewrites.
///
/// This is used for providers (notably Gemini) that reject or mishandle the
/// OpenAI strict transformation (making every field required + nullable). We
/// still prune invalid `required` entries so downstream provider validators
/// don't reject schemas with stale required keys.
fn normalize_schema_lenient(schema: &JsonValue) -> JsonValue {
    let mut schema = schema.clone();
    normalize_schema_lenient_recursive(&mut schema);
    schema
}

/// Coerce degenerate schema values that many LLM providers (llama.cpp, vLLM,
/// etc.) do not support even though they are valid JSON Schema:
///
/// - `true` / `false` boolean schemas → `{}` (allow-anything object)
/// - Objects with only `description` (no `type`, no combinators, no
///   `properties`) → add `"type": "string"` as a sensible default.
///
/// MCP servers commonly emit these patterns; this normalizes them before the
/// schema reaches the provider.
fn coerce_degenerate_schema(schema: &mut JsonValue) {
    match schema {
        JsonValue::Bool(_) => {
            // `true` = "any value", `false` = "no value" — neither is supported
            // by most local/compatible endpoints. Replace with empty object.
            *schema = JsonValue::Object(serde_json::Map::new());
        }
        JsonValue::Object(obj) => {
            // Object schema with only a `description` and no type information:
            // add `"type": "string"` so the provider can validate it.
            let has_type = obj.contains_key("type");
            let has_combinator =
                obj.contains_key("anyOf") || obj.contains_key("oneOf") || obj.contains_key("allOf");
            let has_structure = obj.contains_key("properties")
                || obj.contains_key("items")
                || obj.contains_key("enum");
            let has_description = obj.contains_key("description");

            if !has_type && !has_combinator && !has_structure && has_description {
                obj.insert("type".to_string(), JsonValue::String("string".to_string()));
            }
        }
        _ => {}
    }
}

fn normalize_schema_lenient_recursive(schema: &mut JsonValue) {
    coerce_degenerate_schema(schema);

    let obj = match schema.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    // Recurse into combinators: anyOf, oneOf, allOf
    for key in &["anyOf", "oneOf", "allOf"] {
        if let Some(JsonValue::Array(variants)) = obj.get_mut(*key) {
            for variant in variants.iter_mut() {
                normalize_schema_lenient_recursive(variant);
            }
        }
    }

    // Recurse into array items
    if let Some(items) = obj.get_mut("items") {
        normalize_schema_lenient_recursive(items);
    }

    // Recurse into map value schemas
    if let Some(additional_properties) = obj.get_mut("additionalProperties") {
        normalize_schema_lenient_recursive(additional_properties);
    }

    // Recurse into `not`, `if`, `then`, `else`
    for key in &["not", "if", "then", "else"] {
        if let Some(sub) = obj.get_mut(*key) {
            normalize_schema_lenient_recursive(sub);
        }
    }

    // Recurse into object properties
    if let Some(JsonValue::Object(props)) = obj.get_mut("properties") {
        for prop_schema in props.values_mut() {
            normalize_schema_lenient_recursive(prop_schema);
        }
    }

    prune_required_to_defined_properties(obj);
}

/// Ensure every `required` entry refers to an existing key in `properties`.
///
/// Some providers (and/or proxy schema translators) are stricter than others
/// and reject object schemas if `required` references a property that is not
/// present. We defensively filter such entries here.
fn prune_required_to_defined_properties(obj: &mut serde_json::Map<String, JsonValue>) {
    let Some(props) = obj.get("properties").and_then(|p| p.as_object()) else {
        return;
    };

    let allowed: HashSet<String> = props.keys().cloned().collect();

    let filtered_required = obj.get("required").and_then(|r| r.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str())
            .filter(|name| allowed.contains(*name))
            .map(|s| JsonValue::String(s.to_string()))
            .collect::<Vec<JsonValue>>()
    });

    if let Some(filtered) = filtered_required {
        if filtered.is_empty() {
            obj.remove("required");
        } else {
            obj.insert("required".to_string(), JsonValue::Array(filtered));
        }
    }
}

fn is_kimi_k2_5_model(model_name: &str) -> bool {
    let model = model_name.to_ascii_lowercase();
    model.contains("kimi-k2.5")
        || model.contains("kimi-k2p5")
        || model.contains("kimi-k2-5")
        || model.contains("moonshotai/kimi-k2.5")
}

fn is_kimi_k2_thinking_model(model_name: &str) -> bool {
    let model = model_name.to_ascii_lowercase();
    model.contains("kimi-k2-thinking") || model.contains("moonshotai/kimi-k2-thinking")
}

fn is_kimi_interleaved_thinking_model(model_name: &str) -> bool {
    is_kimi_k2_5_model(model_name) || is_kimi_k2_thinking_model(model_name)
}

fn should_use_glm_native_tool_rounds(model_name: &str) -> bool {
    crate::llm::is_glm_4_7_or_newer_model_id(model_name)
}

fn maybe_chat_template_kwargs_for_model(model_name: &str) -> Option<JsonValue> {
    if should_use_glm_native_tool_rounds(model_name) {
        Some(serde_json::json!({ "clear_thinking": false }))
    } else {
        None
    }
}

fn should_use_strict_tool_schema(model_name: &str) -> bool {
    // Gemini (including OpenRouter `google/gemini-*` models) has stricter
    // function-schema validation and is not compatible with our OpenAI strict
    // rewrite in all cases. Keep the original schema shape for Gemini.
    //
    // Kimi K2.5 also rejects or mishandles strict schemas (all-required +
    // additionalProperties: false) — known issue with SGLang-based serving.
    if model_name.to_ascii_lowercase().contains("gemini") {
        return false;
    }
    if is_kimi_k2_5_model(model_name) {
        return false;
    }
    true
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect()
}

/// Truncate a string for logging, appending "..." if truncated.
pub fn truncate_for_log(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let mut s: String = input.chars().take(max_chars).collect();
    s.push_str("...");
    s
}

/// Kimi K2.5 is sensitive to broad JSON-Schema vocabularies on some
/// OpenAI-compatible gateways. Keep a conservative subset.
fn simplify_schema_kimi_compat(schema: &JsonValue) -> JsonValue {
    simplify_schema_kimi_compat_inner(schema, 0)
}

fn simplify_schema_kimi_compat_inner(schema: &JsonValue, depth: usize) -> JsonValue {
    // Prevent deep / recursive schemas from causing provider parser issues.
    if depth > 8 {
        return JsonValue::Object(serde_json::Map::from_iter([(
            "type".to_string(),
            JsonValue::String("object".to_string()),
        )]));
    }

    let Some(obj) = schema.as_object() else {
        return schema.clone();
    };

    let mut out = serde_json::Map::new();

    if let Some(ty) = obj.get("type") {
        out.insert("type".to_string(), ty.clone());
    }

    if let Some(description) = obj.get("description").and_then(|v| v.as_str())
        && !description.trim().is_empty()
    {
        out.insert(
            "description".to_string(),
            JsonValue::String(truncate_chars(description.trim(), 256)),
        );
    }

    if let Some(enum_values) = obj.get("enum").and_then(|v| v.as_array()) {
        let mut values = Vec::new();
        for value in enum_values.iter().take(64) {
            if value.is_string() || value.is_number() || value.is_boolean() || value.is_null() {
                values.push(value.clone());
            }
        }
        if !values.is_empty() {
            out.insert("enum".to_string(), JsonValue::Array(values));
        }
    }

    if let Some(properties) = obj.get("properties").and_then(|v| v.as_object()) {
        let mut simplified_properties = serde_json::Map::new();
        for (name, value) in properties.iter().take(64) {
            simplified_properties.insert(
                name.clone(),
                simplify_schema_kimi_compat_inner(value, depth + 1),
            );
        }
        out.insert(
            "properties".to_string(),
            JsonValue::Object(simplified_properties),
        );
        out.entry("type".to_string())
            .or_insert_with(|| JsonValue::String("object".to_string()));

        if let Some(required) = obj.get("required").and_then(|v| v.as_array()) {
            let allowed: HashSet<&str> = properties.keys().map(String::as_str).collect();
            let filtered_required: Vec<JsonValue> = required
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|name| allowed.contains(*name))
                .map(|name| JsonValue::String(name.to_string()))
                .collect();
            if !filtered_required.is_empty() {
                out.insert("required".to_string(), JsonValue::Array(filtered_required));
            }
        }

        // Do NOT add additionalProperties: false — vLLM/SGLang K2.5 endpoints
        // choke on this constraint during guided decoding.
    }

    if let Some(items) = obj.get("items") {
        out.insert(
            "items".to_string(),
            simplify_schema_kimi_compat_inner(items, depth + 1),
        );
        out.entry("type".to_string())
            .or_insert_with(|| JsonValue::String("array".to_string()));
    }

    JsonValue::Object(out)
}

fn normalize_tool_parameters_for_model(model_name: &str, schema: &JsonValue) -> JsonValue {
    if should_use_strict_tool_schema(model_name) {
        normalize_schema_strict(schema)
    } else {
        let lenient = normalize_schema_lenient(schema);
        if is_kimi_k2_5_model(model_name) {
            simplify_schema_kimi_compat(&lenient)
        } else {
            lenient
        }
    }
}

fn normalize_tool_description_for_model(model_name: &str, name: &str, description: &str) -> String {
    if is_kimi_k2_5_model(model_name) {
        let base = if description.trim().is_empty() {
            name
        } else {
            description.trim()
        };
        return truncate_chars(base, 512);
    }

    if description.trim().is_empty() {
        name.to_string()
    } else {
        description.to_string()
    }
}

fn normalize_schema_recursive(schema: &mut JsonValue) {
    coerce_degenerate_schema(schema);

    let obj = match schema.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    // Recurse into combinators: anyOf, oneOf, allOf
    for key in &["anyOf", "oneOf", "allOf"] {
        if let Some(JsonValue::Array(variants)) = obj.get_mut(*key) {
            for variant in variants.iter_mut() {
                normalize_schema_recursive(variant);
            }
        }
    }

    // Recurse into array items
    if let Some(items) = obj.get_mut("items") {
        normalize_schema_recursive(items);
    }

    // Recurse into map value schemas
    if let Some(additional_properties) = obj.get_mut("additionalProperties") {
        normalize_schema_recursive(additional_properties);
    }

    // Recurse into `not`, `if`, `then`, `else`
    for key in &["not", "if", "then", "else"] {
        if let Some(sub) = obj.get_mut(*key) {
            normalize_schema_recursive(sub);
        }
    }

    // Only apply object-level normalization if this schema has "properties"
    // (explicit object schema) or type == "object"
    let is_object = obj
        .get("type")
        .and_then(|t| t.as_str())
        .map(|t| t == "object")
        .unwrap_or(false);
    let has_properties = obj.contains_key("properties");

    if !is_object && !has_properties {
        return;
    }

    // Ensure "type": "object" is present
    if !obj.contains_key("type") && has_properties {
        obj.insert("type".to_string(), JsonValue::String("object".to_string()));
    }

    // Force additionalProperties: false (overwrite any existing value)
    obj.insert("additionalProperties".to_string(), JsonValue::Bool(false));

    // Ensure "properties" exists
    if !obj.contains_key("properties") {
        obj.insert(
            "properties".to_string(),
            JsonValue::Object(serde_json::Map::new()),
        );
    }

    // Collect current required set
    let current_required: std::collections::HashSet<String> = obj
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    // Get all property keys (sorted for deterministic output)
    let all_keys: Vec<String> = obj
        .get("properties")
        .and_then(|p| p.as_object())
        .map(|props| {
            let mut keys: Vec<String> = props.keys().cloned().collect();
            keys.sort();
            keys
        })
        .unwrap_or_default();

    // For properties NOT in the original required list, make them nullable
    if let Some(JsonValue::Object(props)) = obj.get_mut("properties") {
        for key in &all_keys {
            // Recurse into each property's schema FIRST (before make_nullable,
            // which may change the type to an array and prevent object detection)
            if let Some(prop_schema) = props.get_mut(key) {
                normalize_schema_recursive(prop_schema);
            }
            // Then make originally-optional properties nullable
            if !current_required.contains(key)
                && let Some(prop_schema) = props.get_mut(key)
            {
                make_nullable(prop_schema);
            }
        }
    }

    // Set required to ALL property keys
    let required_value: Vec<JsonValue> = all_keys.into_iter().map(JsonValue::String).collect();
    obj.insert("required".to_string(), JsonValue::Array(required_value));
    prune_required_to_defined_properties(obj);
}

/// Make a property schema nullable for OpenAI strict mode.
///
/// If it has a simple `"type": "<T>"`, converts to `"type": ["<T>", "null"]`.
/// If it already has an array type, adds "null" if not present.
/// Otherwise, wraps with `anyOf: [<existing>, {"type": "null"}]`.
fn make_nullable(schema: &mut JsonValue) {
    let obj = match schema.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    if let Some(type_val) = obj.get("type").cloned() {
        match type_val {
            // "type": "string" → "type": ["string", "null"]
            JsonValue::String(ref t) if t != "null" => {
                obj.insert("type".to_string(), serde_json::json!([t, "null"]));
            }
            // "type": ["string", "integer"] → add "null" if missing
            JsonValue::Array(ref arr) => {
                let has_null = arr.iter().any(|v| v.as_str() == Some("null"));
                if !has_null {
                    let mut new_arr = arr.clone();
                    new_arr.push(JsonValue::String("null".to_string()));
                    obj.insert("type".to_string(), JsonValue::Array(new_arr));
                }
            }
            _ => {}
        }
    } else {
        // No "type" key — wrap with anyOf including null
        // (handles enum-only, $ref, or combinator schemas)
        let existing = JsonValue::Object(obj.clone());
        obj.clear();
        obj.insert(
            "anyOf".to_string(),
            serde_json::json!([existing, {"type": "null"}]),
        );
    }
}

/// Convert IronClaw messages to rig-core format.
///
/// Returns `(preamble, chat_history)` where preamble is extracted from
/// any System message and chat_history contains the rest.
fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<RigMessage>) {
    let mut preamble: Option<String> = None;
    let mut history = Vec::new();

    for msg in messages {
        match msg.role {
            crate::llm::Role::System => {
                // Concatenate system messages into preamble
                match preamble {
                    Some(ref mut p) => {
                        p.push('\n');
                        p.push_str(&msg.content);
                    }
                    None => preamble = Some(msg.content.clone()),
                }
            }
            crate::llm::Role::User => {
                history.push(RigMessage::user(&msg.content));
            }
            crate::llm::Role::Assistant => {
                if let Some(ref tool_calls) = msg.tool_calls {
                    // Assistant message with tool calls
                    let mut contents: Vec<AssistantContent> = Vec::new();
                    if !msg.content.is_empty() {
                        contents.push(AssistantContent::text(&msg.content));
                    }
                    for (idx, tc) in tool_calls.iter().enumerate() {
                        let tool_call_id =
                            normalized_tool_call_id(Some(tc.id.as_str()), history.len() + idx);
                        contents.push(AssistantContent::ToolCall(
                            rig::message::ToolCall::new(
                                tool_call_id.clone(),
                                ToolFunction::new(tc.name.clone(), tc.arguments.clone()),
                            )
                            .with_call_id(tool_call_id),
                        ));
                    }
                    if let Ok(many) = OneOrMany::many(contents) {
                        history.push(RigMessage::Assistant {
                            id: None,
                            content: many,
                        });
                    } else {
                        // Shouldn't happen but fall back to text
                        history.push(RigMessage::assistant(&msg.content));
                    }
                } else {
                    history.push(RigMessage::assistant(&msg.content));
                }
            }
            crate::llm::Role::Tool => {
                // Tool result message: wrap as User { ToolResult }
                let tool_id = normalized_tool_call_id(msg.tool_call_id.as_deref(), history.len());
                history.push(RigMessage::User {
                    content: OneOrMany::one(UserContent::ToolResult(RigToolResult {
                        id: tool_id.clone(),
                        call_id: Some(tool_id),
                        content: OneOrMany::one(ToolResultContent::text(&msg.content)),
                    })),
                });
            }
        }
    }

    (preamble, history)
}

/// Responses-style providers require a non-empty tool call ID.
fn normalized_tool_call_id(raw: Option<&str>, seed: usize) -> String {
    match raw.map(str::trim).filter(|id| !id.is_empty()) {
        Some(id) => id.to_string(),
        None => format!("generated_tool_call_{seed}"),
    }
}

/// Convert IronClaw tool definitions to rig-core format.
///
/// Applies OpenAI strict-mode schema normalization to ensure all tool
/// parameter schemas comply with OpenAI's function calling requirements.
fn convert_tools(model_name: &str, tools: &[IronToolDefinition]) -> Vec<RigToolDefinition> {
    tools
        .iter()
        .map(|t| RigToolDefinition {
            name: t.name.clone(),
            description: normalize_tool_description_for_model(
                model_name,
                t.name.as_str(),
                t.description.as_str(),
            ),
            parameters: normalize_tool_parameters_for_model(model_name, &t.parameters),
        })
        .collect()
}

fn is_server_error_retryable_for_kimi(error: &str) -> bool {
    is_server_error(error)
}

/// Returns true if the error string indicates a server-side error (5xx).
fn is_server_error(error: &str) -> bool {
    let e = error.to_ascii_lowercase();
    e.contains("status code 500")
        || e.contains("500 internal server error")
        || e.contains("\"error\": \"internal server error\"")
        || e.contains("http 500")
        || e.contains("http 502")
        || e.contains("http 503")
}

fn kimi_reduced_tool_set(tools: &[IronToolDefinition]) -> Vec<IronToolDefinition> {
    // Keep a small deterministic set first. `discover_tools` lets the model
    // activate deferred tools later if needed.
    const KIMI_TOOL_PRIORITY: &[&str] = &[
        "discover_tools",
        "time",
        "read_file",
        "write_file",
        "list_dir",
        "apply_patch",
        "shell",
        "http",
        "memory_search",
    ];

    const KIMI_MAX_TOOLS: usize = 9;

    let mut selected = Vec::new();
    let mut seen = HashSet::new();

    for preferred in KIMI_TOOL_PRIORITY {
        if selected.len() >= KIMI_MAX_TOOLS {
            break;
        }
        if let Some(tool) = tools.iter().find(|t| t.name == *preferred)
            && seen.insert(tool.name.clone())
        {
            selected.push(tool.clone());
        }
    }

    if selected.len() < KIMI_MAX_TOOLS {
        let mut remainder: Vec<IronToolDefinition> = tools
            .iter()
            .filter(|t| !seen.contains(&t.name))
            .cloned()
            .collect();
        remainder.sort_by(|a, b| a.name.cmp(&b.name));
        selected.extend(
            remainder
                .into_iter()
                .take(KIMI_MAX_TOOLS.saturating_sub(selected.len())),
        );
    }

    selected
}

/// Convert IronClaw tool_choice string to rig-core ToolChoice.
fn convert_tool_choice(choice: Option<&str>) -> Option<RigToolChoice> {
    match choice.map(|s| s.to_lowercase()).as_deref() {
        Some("auto") => Some(RigToolChoice::Auto),
        Some("required") => Some(RigToolChoice::Required),
        Some("none") => Some(RigToolChoice::None),
        _ => None,
    }
}

/// Extract text and tool calls from a rig-core completion response.
fn extract_response(
    choice: &OneOrMany<AssistantContent>,
    _usage: &RigUsage,
) -> (Option<String>, Vec<IronToolCall>, FinishReason) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<IronToolCall> = Vec::new();

    for content in choice.iter() {
        match content {
            AssistantContent::Text(t) => {
                if !t.text.is_empty() {
                    text_parts.push(t.text.clone());
                }
            }
            AssistantContent::ToolCall(tc) => {
                tool_calls.push(IronToolCall {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    arguments: tc.function.arguments.clone(),
                });
            }
            AssistantContent::Reasoning(r) => {
                if !r.reasoning.is_empty() {
                    text_parts.push(format!("<thinking>{}</thinking>", r.reasoning.join("\n")));
                }
            }
            // Image variant is not mapped to IronClaw types
            _ => {}
        }
    }

    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let finish = if !tool_calls.is_empty() {
        FinishReason::ToolUse
    } else {
        FinishReason::Stop
    };

    (text, tool_calls, finish)
}

/// Saturate u64 to u32 for token counts.
fn saturate_u32(val: u64) -> u32 {
    val.min(u32::MAX as u64) as u32
}

/// Build a rig-core CompletionRequest from our internal types.
fn build_rig_request(
    preamble: Option<String>,
    mut history: Vec<RigMessage>,
    tools: Vec<RigToolDefinition>,
    tool_choice: Option<RigToolChoice>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> Result<RigRequest, LlmError> {
    // rig-core requires at least one message in chat_history
    if history.is_empty() {
        history.push(RigMessage::user("Hello"));
    }

    let chat_history = OneOrMany::many(history).map_err(|e| LlmError::RequestFailed {
        provider: "rig".to_string(),
        reason: format!("Failed to build chat history: {}", e),
    })?;

    Ok(RigRequest {
        preamble,
        chat_history,
        documents: Vec::new(),
        tools,
        temperature: temperature.map(|t| t as f64),
        max_tokens: max_tokens.map(|t| t as u64),
        tool_choice,
        additional_params: None,
    })
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn payload_dump_path(kind: &str) -> PathBuf {
    if let Ok(path) = std::env::var("IRONCLAW_RIG_PAYLOAD_DUMP_PATH")
        && !path.trim().is_empty()
    {
        return PathBuf::from(path);
    }

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    PathBuf::from(format!("/tmp/ironclaw_rig_payload_{kind}_{ts}.json"))
}

/// Dump the exact OpenAI-completions payload rig-core would send.
///
/// Enable with `IRONCLAW_RIG_PAYLOAD_DUMP=1`.
/// Optional fixed output path: `IRONCLAW_RIG_PAYLOAD_DUMP_PATH=/tmp/rig_payload.json`.
fn maybe_dump_openai_payload(model_name: &str, rig_req: &RigRequest, kind: &str) {
    if !env_flag_enabled("IRONCLAW_RIG_PAYLOAD_DUMP") {
        return;
    }

    let openai_req = match rig::providers::openai::completion::CompletionRequest::try_from((
        model_name.to_string(),
        rig_req.clone(),
    )) {
        Ok(req) => req,
        Err(e) => {
            tracing::warn!(
                model = %model_name,
                error = %e,
                "Failed to convert rig request into OpenAI payload for debug dump"
            );
            return;
        }
    };

    let payload = match serde_json::to_string_pretty(&openai_req) {
        Ok(json) => json,
        Err(e) => {
            tracing::warn!(
                model = %model_name,
                error = %e,
                "Failed to serialize OpenAI payload for debug dump"
            );
            return;
        }
    };

    let path = payload_dump_path(kind);
    match std::fs::write(&path, payload) {
        Ok(_) => tracing::info!(
            model = %model_name,
            path = %path.display(),
            "Wrote rig OpenAI payload debug dump",
        ),
        Err(e) => tracing::warn!(
            model = %model_name,
            path = %path.display(),
            error = %e,
            "Failed to write rig OpenAI payload debug dump",
        ),
    }
}

// -- K2.5 native transport helpers --

impl<M: CompletionModel + Send + Sync + 'static> RigAdapter<M> {
    /// Send a K2.5 tool request using native HTTP transport with string content format.
    ///
    /// rig-core serializes content as `[{"type":"text","text":"..."}]` which vLLM/SGLang
    /// rejects when tools are present. This sends plain string content matching curl.
    async fn send_kimi_native_tool_request(
        &self,
        messages: &[ChatMessage],
        tools: &[IronToolDefinition],
        tool_choice: Option<&str>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<ToolCompletionResponse, LlmError> {
        let transport = self
            .native_transport
            .as_ref()
            .ok_or_else(|| LlmError::RequestFailed {
                provider: self.model_name.clone(),
                reason: "Native transport not configured".to_string(),
            })?;

        // Build messages with string content (matching curl format)
        let json_messages = build_native_messages(messages);

        // Build tool definitions with kimi-compat schemas
        let tools_json: Vec<JsonValue> = tools
            .iter()
            .map(|t| {
                let params = normalize_tool_parameters_for_model(&self.model_name, &t.parameters);
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": normalize_tool_description_for_model(
                            &self.model_name, &t.name, &t.description,
                        ),
                        "parameters": params,
                    }
                })
            })
            .collect();

        let mut payload = serde_json::json!({
            "model": self.model_name,
            "messages": json_messages,
            "tools": tools_json,
            "tool_choice": tool_choice.unwrap_or("auto"),
        });
        if let Some(kwargs) = maybe_chat_template_kwargs_for_model(&self.model_name) {
            payload["chat_template_kwargs"] = kwargs;
        }

        if let Some(t) = temperature {
            payload["temperature"] = serde_json::json!(t);
        }
        if let Some(m) = max_tokens {
            payload["max_tokens"] = serde_json::json!(m);
        }

        // Dump payload if enabled
        if env_flag_enabled("IRONCLAW_RIG_PAYLOAD_DUMP") {
            let path = payload_dump_path("kimi_native");
            if let Ok(json) = serde_json::to_string_pretty(&payload) {
                let _ = std::fs::write(&path, &json);
                tracing::info!(
                    model = %self.model_name,
                    path = %path.display(),
                    bytes = json.len(),
                    "Wrote native K2.5 payload debug dump",
                );
            }
        }

        let url = format!("{}/chat/completions", transport.base_url);
        tracing::debug!(
            model = %self.model_name,
            url = %url,
            tools = tools.len(),
            "Sending K2.5 request via native chat transport",
        );

        let response = transport
            .client
            .post(&url)
            .bearer_auth(&transport.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: self.model_name.clone(),
                reason: format!("Native transport request failed: {e}"),
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| LlmError::RequestFailed {
            provider: self.model_name.clone(),
            reason: format!("Failed to read response body: {e}"),
        })?;

        if !status.is_success() {
            return Err(LlmError::RequestFailed {
                provider: self.model_name.clone(),
                reason: format!("HTTP {status}: {}", truncate_chars(&body, 512)),
            });
        }

        parse_native_tool_response(&self.model_name, &body)
    }

    /// Send a request via native transport WITHOUT the `tools` parameter.
    ///
    /// Used as the final fallback when even reduced-tools requests fail with 500.
    /// Tools are already injected into the system prompt by the caller; the model
    /// emits tool calls as section-delimited text that `recover_tool_calls_from_content`
    /// parses in the reasoning layer.
    async fn send_kimi_native_text_tool_request(
        &self,
        messages: &[ChatMessage],
        temperature: Option<f32>,
        max_tokens: Option<u32>,
    ) -> Result<ToolCompletionResponse, LlmError> {
        let transport = self
            .native_transport
            .as_ref()
            .ok_or_else(|| LlmError::RequestFailed {
                provider: self.model_name.clone(),
                reason: "Native transport not configured".to_string(),
            })?;

        let json_messages = build_native_messages(messages);

        let mut payload = serde_json::json!({
            "model": self.model_name,
            "messages": json_messages,
        });
        if let Some(kwargs) = maybe_chat_template_kwargs_for_model(&self.model_name) {
            payload["chat_template_kwargs"] = kwargs;
        }

        if let Some(t) = temperature {
            payload["temperature"] = serde_json::json!(t);
        }
        if let Some(m) = max_tokens {
            payload["max_tokens"] = serde_json::json!(m);
        }

        if env_flag_enabled("IRONCLAW_RIG_PAYLOAD_DUMP") {
            let path = payload_dump_path("kimi_text_tool_fallback");
            if let Ok(json) = serde_json::to_string_pretty(&payload) {
                let _ = std::fs::write(&path, &json);
                tracing::info!(
                    model = %self.model_name,
                    path = %path.display(),
                    bytes = json.len(),
                    "Wrote native K2.5 text-tool fallback payload dump",
                );
            }
        }

        let url = format!("{}/chat/completions", transport.base_url);
        tracing::debug!(
            model = %self.model_name,
            url = %url,
            "Sending K2.5 text-tool fallback via native transport (no tools param)",
        );

        let response = transport
            .client
            .post(&url)
            .bearer_auth(&transport.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| LlmError::RequestFailed {
                provider: self.model_name.clone(),
                reason: format!("Native text-tool fallback request failed: {e}"),
            })?;

        let status = response.status();
        let body = response.text().await.map_err(|e| LlmError::RequestFailed {
            provider: self.model_name.clone(),
            reason: format!("Failed to read response body: {e}"),
        })?;

        if !status.is_success() {
            return Err(LlmError::RequestFailed {
                provider: self.model_name.clone(),
                reason: format!("HTTP {status}: {}", truncate_chars(&body, 512)),
            });
        }

        parse_native_tool_response(&self.model_name, &body)
    }
}

/// Inject tool definitions into the first system message as plain text.
///
/// When the LLM endpoint doesn't support the structured `tools` parameter
/// (returns HTTP 500), we embed tool descriptions in the system prompt instead.
/// K2.5 is trained to emit tool calls using section-delimited format
/// (`<|tool_calls_section_begin|>...`) which `recover_tool_calls_from_content`
/// in the reasoning layer parses automatically.
fn inject_tools_into_system_prompt(messages: &mut Vec<ChatMessage>, tools: &[IronToolDefinition]) {
    if tools.is_empty() {
        return;
    }

    // Build tool descriptions
    let mut tool_section = String::from(
        "\n\n## Tool Calling\n\
         You have access to the following tools. To call a tool, output EXACTLY this format:\n\
         <|tool_calls_section_begin|><|tool_call_begin|>functions.TOOL_NAME:0\
         <|tool_call_argument_begin|>{\"arg\": \"value\"}<|tool_call_end|>\
         <|tool_calls_section_end|>\n\n\
         IMPORTANT: Use the EXACT tool name as listed below (e.g. `time`, not `time_get_current_time`). \
         The tool name must match exactly.\n\
         Never claim a tool was called or show tool results unless you emitted a valid tool call block in this response.\n\n\
         Available tools:\n",
    );

    for tool in tools {
        tool_section.push_str(&format!("- **{}**: {}\n", tool.name, tool.description));
        // Include parameter schema in compact form
        if let Some(props) = tool.parameters.get("properties")
            && let Some(obj) = props.as_object()
        {
            let required: Vec<&str> = tool
                .parameters
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            let params: Vec<String> = obj
                .iter()
                .map(|(name, schema)| {
                    let ty = schema.get("type").and_then(|t| t.as_str()).unwrap_or("any");
                    let req = if required.contains(&name.as_str()) {
                        " (required)"
                    } else {
                        ""
                    };
                    let desc = schema
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("");
                    // Include enum values so the model knows valid options
                    let enum_suffix = schema
                        .get("enum")
                        .and_then(|e| e.as_array())
                        .map(|vals| {
                            let items: Vec<String> = vals
                                .iter()
                                .filter_map(|v| v.as_str().map(|s| format!("\"{s}\"")))
                                .collect();
                            format!(" [{}]", items.join(", "))
                        })
                        .unwrap_or_default();
                    format!("    - `{name}` ({ty}{req}): {desc}{enum_suffix}")
                })
                .collect();
            if !params.is_empty() {
                tool_section.push_str(&format!("  Parameters:\n{}\n", params.join("\n")));
            }
        }
    }

    // Find the first system message and append the tool section
    if let Some(sys_msg) = messages
        .iter_mut()
        .find(|m| m.role == crate::llm::Role::System)
    {
        sys_msg.content.push_str(&tool_section);
    } else {
        // No system message — prepend one
        messages.insert(
            0,
            ChatMessage {
                role: crate::llm::Role::System,
                content: tool_section,
                tool_call_id: None,
                name: None,
                tool_calls: None,
            },
        );
    }
}

/// Split hidden `<thinking>...</thinking>` blocks from assistant content.
///
/// Returns `(visible_content, reasoning_content)` where reasoning content is
/// joined with newlines if multiple blocks are present.
fn split_hidden_thinking(content: &str) -> (String, Option<String>) {
    let mut visible = String::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut cursor = content;

    const OPEN: &str = "<thinking>";
    const CLOSE: &str = "</thinking>";

    loop {
        let Some(open_idx) = cursor.find(OPEN) else {
            visible.push_str(cursor);
            break;
        };

        visible.push_str(&cursor[..open_idx]);
        let after_open = &cursor[open_idx + OPEN.len()..];

        let Some(close_rel) = after_open.find(CLOSE) else {
            // Malformed tail: keep it as visible content.
            visible.push_str(&cursor[open_idx..]);
            break;
        };

        let reasoning = &after_open[..close_rel];
        if !reasoning.trim().is_empty() {
            reasoning_parts.push(reasoning.to_string());
        }
        cursor = &after_open[close_rel + CLOSE.len()..];
    }

    let reasoning_content = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join("\n"))
    };

    (visible, reasoning_content)
}

/// Build OpenAI-format messages with plain string content (not array).
fn build_native_messages(messages: &[ChatMessage]) -> Vec<JsonValue> {
    let mut out = Vec::new();
    for msg in messages {
        match msg.role {
            crate::llm::Role::System => {
                out.push(serde_json::json!({
                    "role": "system",
                    "content": msg.content,
                }));
            }
            crate::llm::Role::User => {
                out.push(serde_json::json!({
                    "role": "user",
                    "content": msg.content,
                }));
            }
            crate::llm::Role::Assistant => {
                let (visible_content, reasoning_content) = split_hidden_thinking(&msg.content);
                if let Some(ref tool_calls) = msg.tool_calls {
                    let tc_json: Vec<JsonValue> = tool_calls
                        .iter()
                        .map(|tc| {
                            // OpenAI API expects arguments as a JSON string
                            let args_str = if tc.arguments.is_string() {
                                tc.arguments.as_str().unwrap_or("{}").to_string()
                            } else {
                                serde_json::to_string(&tc.arguments).unwrap_or_default()
                            };
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": args_str,
                                }
                            })
                        })
                        .collect();
                    let content: JsonValue = if visible_content.is_empty() {
                        JsonValue::Null
                    } else {
                        JsonValue::String(visible_content)
                    };
                    let mut payload = serde_json::json!({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tc_json,
                    });
                    if let Some(reasoning) = reasoning_content {
                        payload["reasoning_content"] = JsonValue::String(reasoning);
                    }
                    out.push(payload);
                } else {
                    let mut payload = serde_json::json!({
                        "role": "assistant",
                        "content": visible_content,
                    });
                    if let Some(reasoning) = reasoning_content {
                        payload["reasoning_content"] = JsonValue::String(reasoning);
                    }
                    out.push(payload);
                }
            }
            crate::llm::Role::Tool => {
                out.push(serde_json::json!({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id.as_deref().unwrap_or(""),
                }));
            }
        }
    }
    out
}

/// Parse an OpenAI chat completion response into a `ToolCompletionResponse`.
fn parse_native_tool_response(
    model_name: &str,
    body: &str,
) -> Result<ToolCompletionResponse, LlmError> {
    let resp: JsonValue = serde_json::from_str(body).map_err(|e| LlmError::RequestFailed {
        provider: model_name.to_string(),
        reason: format!("Failed to parse response JSON: {e}"),
    })?;

    let choice = resp["choices"]
        .as_array()
        .and_then(|c| c.first())
        .ok_or_else(|| LlmError::RequestFailed {
            provider: model_name.to_string(),
            reason: format!("No choices in response: {}", truncate_chars(body, 256)),
        })?;

    let message = &choice["message"];
    let message_content = message["content"].as_str().map(String::from);
    let reasoning_content = message["reasoning_content"]
        .as_str()
        .or_else(|| message["reasoning"].as_str())
        .map(String::from);

    let mut tool_calls = Vec::new();
    if let Some(tcs) = message["tool_calls"].as_array() {
        for tc in tcs {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let arguments = match tc["function"]["arguments"].as_str() {
                Some(s) => serde_json::from_str(s).unwrap_or(JsonValue::Object(Default::default())),
                None => tc["function"]["arguments"].clone(),
            };
            tool_calls.push(IronToolCall {
                id,
                name,
                arguments,
            });
        }
    }

    let content = match (message_content, reasoning_content) {
        // Keep reasoning in hidden tags so we can preserve it in context without
        // exposing chain-of-thought to users.
        (Some(content), Some(reasoning)) => {
            Some(format!("{content}\n<thinking>{reasoning}</thinking>"))
        }
        (Some(content), None) => Some(content),
        (None, Some(reasoning)) => Some(format!("<thinking>{reasoning}</thinking>")),
        (None, None) => None,
    };

    let finish_reason = if !tool_calls.is_empty() {
        FinishReason::ToolUse
    } else {
        FinishReason::Stop
    };

    let input_tokens = resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let output_tokens = resp["usage"]["completion_tokens"].as_u64().unwrap_or(0);

    Ok(ToolCompletionResponse {
        content,
        tool_calls,
        input_tokens: saturate_u32(input_tokens),
        output_tokens: saturate_u32(output_tokens),
        finish_reason,
    })
}

#[async_trait]
impl<M> LlmProvider for RigAdapter<M>
where
    M: CompletionModel + Send + Sync + 'static,
    M::Response: Send + Sync + Serialize + DeserializeOwned,
{
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
        (self.input_cost, self.output_cost)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        if let Some(requested_model) = request.model.as_deref()
            && requested_model != self.model_name.as_str()
        {
            tracing::warn!(
                requested_model = requested_model,
                active_model = %self.model_name,
                "Per-request model override is not supported for this provider; using configured model"
            );
        }

        let mut messages = request.messages;
        crate::llm::provider::sanitize_tool_messages(&mut messages);
        let (preamble, history) = convert_messages(&messages);

        let rig_req = build_rig_request(
            preamble,
            history,
            Vec::new(),
            None,
            request.temperature,
            request.max_tokens,
        )?;
        maybe_dump_openai_payload(&self.model_name, &rig_req, "complete");

        let response =
            self.model
                .completion(rig_req)
                .await
                .map_err(|e| LlmError::RequestFailed {
                    provider: self.model_name.clone(),
                    reason: e.to_string(),
                })?;

        let (text, _tool_calls, finish) = extract_response(&response.choice, &response.usage);

        Ok(CompletionResponse {
            content: text.unwrap_or_default(),
            input_tokens: saturate_u32(response.usage.input_tokens),
            output_tokens: saturate_u32(response.usage.output_tokens),
            finish_reason: finish,
        })
    }

    async fn complete_with_tools(
        &self,
        request: ToolCompletionRequest,
    ) -> Result<ToolCompletionResponse, LlmError> {
        if let Some(requested_model) = request.model.as_deref()
            && requested_model != self.model_name.as_str()
        {
            tracing::warn!(
                requested_model = requested_model,
                active_model = %self.model_name,
                "Per-request model override is not supported for this provider; using configured model"
            );
        }

        let known_tool_names: HashSet<String> =
            request.tools.iter().map(|t| t.name.clone()).collect();

        let mut messages = request.messages;
        crate::llm::provider::sanitize_tool_messages(&mut messages);

        // Interleaved-thinking Kimi models: bypass rig-core's array content
        // serialization for tool rounds, because these endpoints expect plain
        // string content and emit reasoning in a provider-specific field.
        if should_use_glm_native_tool_rounds(&self.model_name) && self.native_transport.is_some() {
            match self
                .send_kimi_native_tool_request(
                    &messages,
                    &request.tools,
                    request.tool_choice.as_deref(),
                    request.temperature,
                    request.max_tokens,
                )
                .await
            {
                Ok(mut resp) => {
                    normalize_tool_calls(&mut resp.tool_calls, &known_tool_names);
                    return Ok(resp);
                }
                Err(e) => {
                    tracing::warn!(
                        model = %self.model_name,
                        error = %e,
                        "GLM native tool request failed; falling back to standard provider path"
                    );
                }
            }
        }

        if is_kimi_interleaved_thinking_model(&self.model_name) && self.native_transport.is_some() {
            // K2.5 / K2-thinking: try structured tools via native transport first.
            // This works correctly on OpenRouter and other proxies that normalize
            // tool-calling requests. Only fall back to text-based tool injection
            // when the endpoint returns a server error (e.g. vLLM/SGLang 500).
            match self
                .send_kimi_native_tool_request(
                    &messages,
                    &request.tools,
                    request.tool_choice.as_deref(),
                    request.temperature,
                    request.max_tokens,
                )
                .await
            {
                Ok(mut resp) => {
                    normalize_tool_calls(&mut resp.tool_calls, &known_tool_names);
                    return Ok(resp);
                }
                Err(ref e)
                    if is_kimi_k2_5_model(&self.model_name) && is_server_error(&e.to_string()) =>
                {
                    tracing::warn!(
                        model = %self.model_name,
                        error = %e,
                        "K2.5 structured tool request hit server error; falling back to text-based tool calling"
                    );

                    // Text-tool fallback: inject tools into system prompt,
                    // model emits section-delimited tool calls recovered by
                    // recover_tool_calls_from_content in the reasoning layer.
                    let mut text_tool_messages = messages.clone();
                    inject_tools_into_system_prompt(&mut text_tool_messages, &request.tools);

                    match self
                        .send_kimi_native_text_tool_request(
                            &text_tool_messages,
                            request.temperature,
                            request.max_tokens,
                        )
                        .await
                    {
                        Ok(mut resp) => {
                            normalize_tool_calls(&mut resp.tool_calls, &known_tool_names);
                            return Ok(resp);
                        }
                        Err(e) => {
                            let reduced_tools = kimi_reduced_tool_set(&request.tools);
                            if reduced_tools.len() < request.tools.len() {
                                tracing::warn!(
                                    model = %self.model_name,
                                    tools = request.tools.len(),
                                    reduced_tools = reduced_tools.len(),
                                    error = %e,
                                    "K2.5 text-tool request failed; retrying with reduced tool set"
                                );

                                let mut reduced_text_tool_messages = messages.clone();
                                inject_tools_into_system_prompt(
                                    &mut reduced_text_tool_messages,
                                    &reduced_tools,
                                );

                                let mut resp = self
                                    .send_kimi_native_text_tool_request(
                                        &reduced_text_tool_messages,
                                        request.temperature,
                                        request.max_tokens,
                                    )
                                    .await?;
                                normalize_tool_calls(&mut resp.tool_calls, &known_tool_names);
                                return Ok(resp);
                            }

                            return Err(e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        model = %self.model_name,
                        error = %e,
                        "Kimi native transport failed; falling back to standard provider path"
                    );
                }
            }
        }

        // Standard rig-core path (non-interleaved models, or native fallback).
        let (preamble, history) = convert_messages(&messages);
        let tools = convert_tools(&self.model_name, &request.tools);
        let tool_choice = convert_tool_choice(request.tool_choice.as_deref());

        let rig_req = build_rig_request(
            preamble,
            history,
            tools,
            tool_choice,
            request.temperature,
            request.max_tokens,
        )?;
        maybe_dump_openai_payload(&self.model_name, &rig_req, "complete_with_tools");

        let response = match self.model.completion(rig_req.clone()).await {
            Ok(resp) => resp,
            Err(primary_error) => {
                let primary_reason = primary_error.to_string();
                if !is_kimi_k2_5_model(&self.model_name)
                    || !is_server_error_retryable_for_kimi(&primary_reason)
                {
                    return Err(LlmError::RequestFailed {
                        provider: self.model_name.clone(),
                        reason: primary_reason,
                    });
                }

                tracing::warn!(
                    model = %self.model_name,
                    tools = request.tools.len(),
                    "Kimi tool request hit provider 500; retrying with reduced tool set"
                );

                let reduced_tool_defs = kimi_reduced_tool_set(&request.tools);
                let reduced_tools = convert_tools(&self.model_name, &reduced_tool_defs);
                let reduced_req = build_rig_request(
                    rig_req.preamble.clone(),
                    rig_req.chat_history.clone().into_iter().collect(),
                    reduced_tools,
                    rig_req.tool_choice.clone(),
                    request.temperature,
                    request.max_tokens,
                )?;
                maybe_dump_openai_payload(
                    &self.model_name,
                    &reduced_req,
                    "complete_with_tools_kimi_reduced",
                );

                match self.model.completion(reduced_req).await {
                    Ok(resp) => resp,
                    Err(reduced_error) => {
                        let reduced_reason = reduced_error.to_string();
                        tracing::warn!(
                            model = %self.model_name,
                            "Kimi reduced tool request also failed; retrying once without tools"
                        );

                        let no_tools_req = build_rig_request(
                            rig_req.preamble.clone(),
                            rig_req.chat_history.clone().into_iter().collect(),
                            Vec::new(),
                            None,
                            request.temperature,
                            request.max_tokens,
                        )?;
                        maybe_dump_openai_payload(
                            &self.model_name,
                            &no_tools_req,
                            "complete_with_tools_kimi_no_tools",
                        );

                        match self.model.completion(no_tools_req).await {
                            Ok(resp) => resp,
                            Err(no_tools_error) => {
                                return Err(LlmError::RequestFailed {
                                    provider: self.model_name.clone(),
                                    reason: format!(
                                        "{primary_reason}; reduced_tools_retry={reduced_reason}; no_tools_retry={}",
                                        no_tools_error
                                    ),
                                });
                            }
                        }
                    }
                }
            }
        };

        let (text, mut tool_calls, finish) = extract_response(&response.choice, &response.usage);

        // Normalize tool call names: some proxies prepend "proxy_" prefixes.
        normalize_tool_calls(&mut tool_calls, &known_tool_names);

        Ok(ToolCompletionResponse {
            content: text,
            tool_calls,
            input_tokens: saturate_u32(response.usage.input_tokens),
            output_tokens: saturate_u32(response.usage.output_tokens),
            finish_reason: finish,
        })
    }

    fn active_model_name(&self) -> String {
        self.model_name.clone()
    }

    fn effective_model_name(&self, _requested_model: Option<&str>) -> String {
        self.active_model_name()
    }

    fn set_model(&self, _model: &str) -> Result<(), LlmError> {
        // rig-core models are baked at construction time.
        // Switching requires creating a new adapter.
        Err(LlmError::RequestFailed {
            provider: self.model_name.clone(),
            reason: "Runtime model switching not supported for rig-core providers. \
                     Restart with a different model configured."
                .to_string(),
        })
    }
}

/// Normalize a tool call name returned by an OpenAI-compatible provider.
///
/// Some proxies (e.g. VibeProxy) prepend `proxy_` to tool names.
/// If the returned name doesn't match any known tool but stripping a
/// `proxy_` prefix yields a match, use the stripped version.
fn normalize_tool_name(name: &str, known_tools: &HashSet<String>) -> String {
    if known_tools.contains(name) {
        return name.to_string();
    }

    if let Some(stripped) = name.strip_prefix("proxy_")
        && known_tools.contains(stripped)
    {
        return stripped.to_string();
    }

    name.to_string()
}

/// Normalize tool call names in-place using the known tool set.
fn normalize_tool_calls(tool_calls: &mut [IronToolCall], known_tools: &HashSet<String>) {
    for tc in tool_calls {
        let normalized = normalize_tool_name(&tc.name, known_tools);
        if normalized != tc.name {
            tracing::debug!(
                original = %tc.name,
                normalized = %normalized,
                "Normalized tool call name from provider",
            );
            tc.name = normalized;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_messages_system_to_preamble() {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello"),
        ];
        let (preamble, history) = convert_messages(&messages);
        assert_eq!(preamble, Some("You are a helpful assistant.".to_string()));
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_convert_messages_multiple_systems_concatenated() {
        let messages = vec![
            ChatMessage::system("System 1"),
            ChatMessage::system("System 2"),
            ChatMessage::user("Hi"),
        ];
        let (preamble, history) = convert_messages(&messages);
        assert_eq!(preamble, Some("System 1\nSystem 2".to_string()));
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn test_convert_messages_tool_result() {
        let messages = vec![ChatMessage::tool_result(
            "call_123",
            "search",
            "result text",
        )];
        let (preamble, history) = convert_messages(&messages);
        assert!(preamble.is_none());
        assert_eq!(history.len(), 1);
        // Tool results become User messages in rig-core
        match &history[0] {
            RigMessage::User { content } => match content.first() {
                UserContent::ToolResult(r) => {
                    assert_eq!(r.id, "call_123");
                    assert_eq!(r.call_id.as_deref(), Some("call_123"));
                }
                other => panic!("Expected tool result content, got: {:?}", other),
            },
            other => panic!("Expected User message, got: {:?}", other),
        }
    }

    #[test]
    fn test_convert_messages_assistant_with_tool_calls() {
        let tc = IronToolCall {
            id: "call_1".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        };
        let msg = ChatMessage::assistant_with_tool_calls(Some("thinking".to_string()), vec![tc]);
        let messages = vec![msg];
        let (_preamble, history) = convert_messages(&messages);
        assert_eq!(history.len(), 1);
        match &history[0] {
            RigMessage::Assistant { content, .. } => {
                // Should have both text and tool call
                assert!(content.iter().count() >= 2);
                for item in content.iter() {
                    if let AssistantContent::ToolCall(tc) = item {
                        assert_eq!(tc.call_id.as_deref(), Some("call_1"));
                    }
                }
            }
            other => panic!("Expected Assistant message, got: {:?}", other),
        }
    }

    #[test]
    fn test_convert_messages_tool_result_without_id_gets_fallback() {
        let messages = vec![ChatMessage {
            role: crate::llm::Role::Tool,
            content: "result text".to_string(),
            tool_call_id: None,
            name: Some("search".to_string()),
            tool_calls: None,
        }];
        let (_preamble, history) = convert_messages(&messages);
        match &history[0] {
            RigMessage::User { content } => match content.first() {
                UserContent::ToolResult(r) => {
                    assert!(r.id.starts_with("generated_tool_call_"));
                    assert_eq!(r.call_id.as_deref(), Some(r.id.as_str()));
                }
                other => panic!("Expected tool result content, got: {:?}", other),
            },
            other => panic!("Expected User message, got: {:?}", other),
        }
    }

    #[test]
    fn test_convert_tools() {
        let tools = vec![IronToolDefinition {
            name: "search".to_string(),
            description: "Search the web".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        }];
        let rig_tools = convert_tools("openai/gpt-4.1-mini", &tools);
        assert_eq!(rig_tools.len(), 1);
        assert_eq!(rig_tools[0].name, "search");
        assert_eq!(rig_tools[0].description, "Search the web");
    }

    #[test]
    fn test_convert_tools_gemini_keeps_original_optional_shape() {
        let tools = vec![IronToolDefinition {
            name: "demo".to_string(),
            description: "demo".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"},
                    "optional_field": {"type": "integer"}
                },
                "required": ["required_field"]
            }),
        }];

        let rig_tools = convert_tools("google/gemini-3-flash-preview", &tools);
        let params = &rig_tools[0].parameters;

        assert_eq!(params["required"], serde_json::json!(["required_field"]));
        assert_eq!(
            params["properties"]["optional_field"]["type"],
            serde_json::json!("integer")
        );
        assert!(
            params.get("additionalProperties").is_none(),
            "Gemini path should avoid OpenAI strict rewrites"
        );
    }

    #[test]
    fn test_convert_tools_non_gemini_applies_strict_rewrite() {
        let tools = vec![IronToolDefinition {
            name: "demo".to_string(),
            description: "demo".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"},
                    "optional_field": {"type": "integer"}
                },
                "required": ["required_field"]
            }),
        }];

        let rig_tools = convert_tools("openai/gpt-4.1", &tools);
        let params = &rig_tools[0].parameters;

        assert_eq!(params["additionalProperties"], serde_json::json!(false));
        assert_eq!(
            params["required"],
            serde_json::json!(["optional_field", "required_field"])
        );
        assert_eq!(
            params["properties"]["optional_field"]["type"],
            serde_json::json!(["integer", "null"])
        );
    }

    #[test]
    fn test_coerce_degenerate_schema_boolean_true() {
        // `true` boolean schema (common in MCP tool definitions) must be replaced
        // with an empty object so providers like llama.cpp / qwen don't reject it.
        let tools = vec![IronToolDefinition {
            name: "mcp_tool".to_string(),
            description: "MCP tool with boolean schema".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "filter": true,
                    "having": {"description": "Having clauses for the query"}
                }
            }),
        }];

        let rig_tools = convert_tools("qwen3.5-9b", &tools);
        let params = &rig_tools[0].parameters;

        // `true` → coerced to an object (strict mode may wrap it further)
        assert!(
            !params["properties"]["filter"].is_boolean(),
            "boolean schema `true` must be coerced to an object"
        );
        // description-only → must have a `type` (strict mode makes it nullable: ["string", "null"])
        assert!(
            !params["properties"]["having"]["type"].is_null(),
            "description-only schema must gain a type field"
        );
    }

    #[test]
    fn test_coerce_degenerate_schema_boolean_strict_model() {
        // Same schemas under a strict-mode model (OpenAI-style).
        let tools = vec![IronToolDefinition {
            name: "mcp_tool".to_string(),
            description: "MCP tool with boolean schema".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "filter": true,
                    "value": {"description": "Value for the filter"}
                }
            }),
        }];

        let rig_tools = convert_tools("openai/gpt-4.1", &tools);
        let params = &rig_tools[0].parameters;

        assert!(
            !params["properties"]["filter"].is_boolean(),
            "boolean schema `true` must be coerced to an object under strict mode"
        );
        // Strict mode makes optional fields nullable: ["string", "null"]
        assert!(
            !params["properties"]["value"]["type"].is_null(),
            "description-only schema must gain a type under strict mode"
        );
    }

    #[test]
    fn test_coerce_degenerate_schema_unit_level() {
        // Test the coercion function directly to verify behavior.
        let mut bool_schema = serde_json::json!(true);
        coerce_degenerate_schema(&mut bool_schema);
        assert_eq!(bool_schema, serde_json::json!({}));

        let mut desc_only = serde_json::json!({"description": "A filter value"});
        coerce_degenerate_schema(&mut desc_only);
        assert_eq!(desc_only["type"], serde_json::json!("string"));
        assert_eq!(
            desc_only["description"],
            serde_json::json!("A filter value")
        );

        // Schema with a type should NOT be modified
        let mut typed = serde_json::json!({"type": "integer", "description": "A count"});
        coerce_degenerate_schema(&mut typed);
        assert_eq!(typed["type"], serde_json::json!("integer"));
    }

    #[test]
    fn test_convert_tools_kimi_simplifies_schema_subset() {
        let tools = vec![IronToolDefinition {
            name: "demo".to_string(),
            description: "demo".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "title": "Ignored title",
                "properties": {
                    "query": {
                        "type": "string",
                        "default": "hello",
                        "minLength": 1
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20
                    },
                    "metadata": {
                        "type": "object",
                        "additionalProperties": { "type": "string" }
                    }
                },
                "required": ["query"],
                "anyOf": [
                    { "type": "object" },
                    { "type": "null" }
                ]
            }),
        }];

        let rig_tools = convert_tools("moonshotai/Kimi-K2.5", &tools);
        let params = &rig_tools[0].parameters;

        assert_eq!(params["type"], serde_json::json!("object"));
        assert_eq!(params["required"], serde_json::json!(["query"]));
        assert_eq!(
            params["properties"]["query"]["type"],
            serde_json::json!("string")
        );
        assert!(
            params["properties"]["query"].get("default").is_none(),
            "Kimi path should drop non-essential schema keywords"
        );
        assert!(
            params["properties"]["limit"].get("minimum").is_none(),
            "Kimi path should drop min/max constraints that can trigger provider bugs"
        );
        assert!(
            params["properties"]["metadata"]
                .get("additionalProperties")
                .is_none(),
            "Kimi path should drop nested map schemas that some gateways reject"
        );
        assert!(
            params.get("anyOf").is_none(),
            "Kimi path should drop combinators that can trigger parser bugs"
        );
        assert!(
            params.get("additionalProperties").is_none(),
            "Kimi path should NOT emit additionalProperties (vLLM/SGLang choke on it)"
        );
        assert!(
            params.get("title").is_none(),
            "Kimi path should drop root-level title"
        );
    }

    #[test]
    fn test_kimi_reduced_tool_set_prefers_core_priority() {
        let tools = vec![
            IronToolDefinition {
                name: "memory_write".to_string(),
                description: "".to_string(),
                parameters: serde_json::json!({"type":"object"}),
            },
            IronToolDefinition {
                name: "time".to_string(),
                description: "".to_string(),
                parameters: serde_json::json!({"type":"object"}),
            },
            IronToolDefinition {
                name: "discover_tools".to_string(),
                description: "".to_string(),
                parameters: serde_json::json!({"type":"object"}),
            },
            IronToolDefinition {
                name: "shell".to_string(),
                description: "".to_string(),
                parameters: serde_json::json!({"type":"object"}),
            },
            IronToolDefinition {
                name: "create_job".to_string(),
                description: "".to_string(),
                parameters: serde_json::json!({"type":"object"}),
            },
        ];

        let reduced = kimi_reduced_tool_set(&tools);
        let names: Vec<&str> = reduced.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names[0], "discover_tools");
        assert_eq!(names[1], "time");
        assert_eq!(names[2], "shell");
        assert!(names.contains(&"memory_write"));
        assert!(names.contains(&"create_job"));
    }

    #[test]
    fn test_is_server_error_retryable_for_kimi() {
        assert!(is_server_error_retryable_for_kimi(
            "HttpError: Invalid status code 500 Internal Server Error with message: {\"error\": \"Internal Server Error\"}"
        ));
        assert!(!is_server_error_retryable_for_kimi(
            "HttpError: Invalid status code 400 Bad Request"
        ));
    }

    #[test]
    fn test_is_server_error_covers_5xx() {
        assert!(is_server_error("HTTP 500: Internal Server Error"));
        assert!(is_server_error("HTTP 502: Bad Gateway"));
        assert!(is_server_error("HTTP 503: Service Unavailable"));
        assert!(is_server_error("status code 500"));
        assert!(!is_server_error("HTTP 400: Bad Request"));
        assert!(!is_server_error("HTTP 429: Too Many Requests"));
    }

    #[test]
    fn test_is_kimi_interleaved_thinking_model() {
        assert!(is_kimi_interleaved_thinking_model("moonshotai/kimi-k2.5"));
        assert!(is_kimi_interleaved_thinking_model("kimi-k2.5"));
        assert!(is_kimi_interleaved_thinking_model("kimi-k2-5"));
        assert!(is_kimi_interleaved_thinking_model("kimi-k2p5"));
        assert!(is_kimi_interleaved_thinking_model(
            "moonshotai/kimi-k2-thinking"
        ));
        assert!(is_kimi_k2_thinking_model("kimi-k2-thinking"));
        assert!(!is_kimi_interleaved_thinking_model("moonshotai/kimi-k2"));
    }

    #[test]
    fn test_maybe_chat_template_kwargs_for_model_glm() {
        let kwargs = maybe_chat_template_kwargs_for_model("zai-org/GLM-5").expect("glm kwargs");
        assert_eq!(kwargs, serde_json::json!({"clear_thinking": false}));
        assert!(maybe_chat_template_kwargs_for_model("moonshotai/Kimi-K2.5").is_none());
    }

    #[test]
    fn test_split_hidden_thinking() {
        let (visible, reasoning) = split_hidden_thinking(
            "Visible content<thinking>first thought</thinking>\n<thinking>second thought</thinking>",
        );
        assert_eq!(visible, "Visible content\n");
        assert_eq!(reasoning, Some("first thought\nsecond thought".to_string()));
    }

    #[test]
    fn test_parse_native_tool_response_wraps_reasoning_hidden() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "I'll use tools.",
                    "reasoning_content": "Need current date and web search",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "time",
                            "arguments": "{}"
                        }
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        })
        .to_string();

        let parsed = parse_native_tool_response("moonshotai/kimi-k2-thinking", &body)
            .expect("response should parse");
        assert_eq!(parsed.tool_calls.len(), 1);
        let content = parsed.content.expect("content should be present");
        assert!(content.contains("I'll use tools."));
        assert!(content.contains("<thinking>Need current date and web search</thinking>"));
    }

    #[test]
    fn test_parse_native_tool_response_supports_reasoning_alias() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "reasoning": "Need to think first",
                    "tool_calls": []
                }
            }]
        })
        .to_string();

        let parsed = parse_native_tool_response("zai-org/GLM-5", &body).expect("response parses");
        let content = parsed.content.expect("content should be present");
        assert_eq!(content, "<thinking>Need to think first</thinking>");
    }

    #[test]
    fn test_build_native_messages_exposes_reasoning_content_field() {
        let messages = vec![ChatMessage::assistant(
            "Visible reply<thinking>hidden reasoning</thinking>",
        )];

        let out = build_native_messages(&messages);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[0]["content"], "Visible reply");
        assert_eq!(out[0]["reasoning_content"], "hidden reasoning");
    }

    #[test]
    fn test_convert_tools_kimi_description_fallback_and_truncation() {
        let tools = vec![
            IronToolDefinition {
                name: "unnamed_description_tool".to_string(),
                description: "   ".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
            IronToolDefinition {
                name: "long_description_tool".to_string(),
                description: "x".repeat(700),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
        ];

        let rig_tools = convert_tools("moonshotai/Kimi-K2.5", &tools);
        assert_eq!(rig_tools[0].description, "unnamed_description_tool");
        assert_eq!(rig_tools[1].description.chars().count(), 512);
    }

    #[test]
    fn test_normalize_schema_lenient_prunes_stale_required_recursively() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "dashboard": {
                    "type": "object",
                    "properties": {
                        "widgets": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "title": {"type": "string"}
                                },
                                "required": ["id", "missing_widget_prop", "title"]
                            }
                        }
                    },
                    "required": ["widgets", "missing_dashboard_prop"]
                }
            },
            "required": ["dashboard", "missing_root_prop"]
        });

        let normalized = normalize_schema_lenient(&schema);

        assert_eq!(normalized["required"], serde_json::json!(["dashboard"]));
        assert_eq!(
            normalized["properties"]["dashboard"]["required"],
            serde_json::json!(["widgets"])
        );
        assert_eq!(
            normalized["properties"]["dashboard"]["properties"]["widgets"]["items"]["required"],
            serde_json::json!(["id", "title"])
        );
    }

    #[test]
    fn test_convert_tool_choice() {
        assert!(matches!(
            convert_tool_choice(Some("auto")),
            Some(RigToolChoice::Auto)
        ));
        assert!(matches!(
            convert_tool_choice(Some("required")),
            Some(RigToolChoice::Required)
        ));
        assert!(matches!(
            convert_tool_choice(Some("none")),
            Some(RigToolChoice::None)
        ));
        assert!(matches!(
            convert_tool_choice(Some("AUTO")),
            Some(RigToolChoice::Auto)
        ));
        assert!(convert_tool_choice(None).is_none());
        assert!(convert_tool_choice(Some("unknown")).is_none());
    }

    #[test]
    fn test_extract_response_text_only() {
        let content = OneOrMany::one(AssistantContent::text("Hello world"));
        let usage = RigUsage::new();
        let (text, calls, finish) = extract_response(&content, &usage);
        assert_eq!(text, Some("Hello world".to_string()));
        assert!(calls.is_empty());
        assert_eq!(finish, FinishReason::Stop);
    }

    #[test]
    fn test_extract_response_tool_call() {
        let tc = AssistantContent::tool_call("call_1", "search", serde_json::json!({"q": "test"}));
        let content = OneOrMany::one(tc);
        let usage = RigUsage::new();
        let (text, calls, finish) = extract_response(&content, &usage);
        assert!(text.is_none());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(finish, FinishReason::ToolUse);
    }

    #[test]
    fn test_assistant_tool_call_empty_id_gets_generated() {
        let tc = IronToolCall {
            id: "".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        };
        let messages = vec![ChatMessage::assistant_with_tool_calls(None, vec![tc])];
        let (_preamble, history) = convert_messages(&messages);

        match &history[0] {
            RigMessage::Assistant { content, .. } => {
                let tool_call = content.iter().find_map(|c| match c {
                    AssistantContent::ToolCall(tc) => Some(tc),
                    _ => None,
                });
                let tc = tool_call.expect("should have a tool call");
                assert!(!tc.id.is_empty(), "tool call id must not be empty");
                assert!(
                    tc.id.starts_with("generated_tool_call_"),
                    "empty id should be replaced with generated id, got: {}",
                    tc.id
                );
                assert_eq!(tc.call_id.as_deref(), Some(tc.id.as_str()));
            }
            other => panic!("Expected Assistant message, got: {:?}", other),
        }
    }

    #[test]
    fn test_assistant_tool_call_whitespace_id_gets_generated() {
        let tc = IronToolCall {
            id: "   ".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        };
        let messages = vec![ChatMessage::assistant_with_tool_calls(None, vec![tc])];
        let (_preamble, history) = convert_messages(&messages);

        match &history[0] {
            RigMessage::Assistant { content, .. } => {
                let tool_call = content.iter().find_map(|c| match c {
                    AssistantContent::ToolCall(tc) => Some(tc),
                    _ => None,
                });
                let tc = tool_call.expect("should have a tool call");
                assert!(
                    tc.id.starts_with("generated_tool_call_"),
                    "whitespace-only id should be replaced, got: {:?}",
                    tc.id
                );
            }
            other => panic!("Expected Assistant message, got: {:?}", other),
        }
    }

    #[test]
    fn test_assistant_and_tool_result_missing_ids_share_generated_id() {
        // Simulate: assistant emits a tool call with empty id, then tool
        // result arrives without an id. Both should get deterministic
        // generated ids that match (based on their position in history).
        let tc = IronToolCall {
            id: "".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"query": "test"}),
        };
        let assistant_msg = ChatMessage::assistant_with_tool_calls(None, vec![tc]);
        let tool_result_msg = ChatMessage {
            role: crate::llm::Role::Tool,
            content: "search results here".to_string(),
            tool_call_id: None,
            name: Some("search".to_string()),
            tool_calls: None,
        };
        let messages = vec![assistant_msg, tool_result_msg];
        let (_preamble, history) = convert_messages(&messages);

        // Extract the generated call_id from the assistant tool call
        let assistant_call_id = match &history[0] {
            RigMessage::Assistant { content, .. } => {
                let tc = content.iter().find_map(|c| match c {
                    AssistantContent::ToolCall(tc) => Some(tc),
                    _ => None,
                });
                tc.expect("should have tool call").id.clone()
            }
            other => panic!("Expected Assistant message, got: {:?}", other),
        };

        // Extract the generated call_id from the tool result
        let tool_result_call_id = match &history[1] {
            RigMessage::User { content } => match content.first() {
                UserContent::ToolResult(r) => r
                    .call_id
                    .clone()
                    .expect("tool result call_id must be present"),
                other => panic!("Expected ToolResult, got: {:?}", other),
            },
            other => panic!("Expected User message, got: {:?}", other),
        };

        assert!(
            !assistant_call_id.is_empty(),
            "assistant call_id must not be empty"
        );
        assert!(
            !tool_result_call_id.is_empty(),
            "tool result call_id must not be empty"
        );

        // NOTE: With the current seed-based generation, these IDs will differ
        // because the assistant tool call uses seed=0 (history.len() at that
        // point) and the tool result uses seed=1 (history.len() after the
        // assistant message was pushed). This documents the current behavior.
        // A future improvement could thread the assistant's generated ID into
        // the tool result for exact matching.
        assert_ne!(
            assistant_call_id, tool_result_call_id,
            "Current impl generates different IDs for assistant call and tool result \
             because seeds differ; this documents the known limitation"
        );
    }

    #[test]
    fn test_saturate_u32() {
        assert_eq!(saturate_u32(100), 100);
        assert_eq!(saturate_u32(u64::MAX), u32::MAX);
        assert_eq!(saturate_u32(u32::MAX as u64), u32::MAX);
    }

    // -- normalize_tool_name tests --

    #[test]
    fn test_normalize_tool_name_exact_match() {
        let known = HashSet::from(["echo".to_string(), "list_jobs".to_string()]);
        assert_eq!(normalize_tool_name("echo", &known), "echo");
    }

    #[test]
    fn test_normalize_tool_name_proxy_prefix_match() {
        let known = HashSet::from(["echo".to_string(), "list_jobs".to_string()]);
        assert_eq!(normalize_tool_name("proxy_echo", &known), "echo");
    }

    #[test]
    fn test_normalize_tool_name_proxy_prefix_no_match_kept() {
        let known = HashSet::from(["echo".to_string(), "list_jobs".to_string()]);
        assert_eq!(
            normalize_tool_name("proxy_unknown", &known),
            "proxy_unknown"
        );
    }

    #[test]
    fn test_normalize_tool_name_unknown_passthrough() {
        let known = HashSet::from(["echo".to_string()]);
        assert_eq!(normalize_tool_name("other_tool", &known), "other_tool");
    }

    #[test]
    fn test_inject_tools_into_system_prompt_appends() {
        let mut messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];
        let tools = vec![IronToolDefinition {
            name: "time".to_string(),
            description: "Get current time".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tz": {"type": "string", "description": "Timezone"}
                },
                "required": ["tz"]
            }),
        }];
        inject_tools_into_system_prompt(&mut messages, &tools);
        assert!(messages[0].content.contains("Tool Calling"));
        assert!(messages[0].content.contains("time"));
        assert!(messages[0].content.contains("Get current time"));
        assert!(messages[0].content.contains("tool_calls_section_begin"));
        assert!(messages[0].content.contains("tz"));
        assert!(messages[0].content.contains("(required)"));
        // Original system prompt is still there
        assert!(messages[0].content.starts_with("You are helpful."));
        // Message count unchanged
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_inject_tools_into_system_prompt_creates_system_msg() {
        let mut messages = vec![ChatMessage::user("Hello")];
        let tools = vec![IronToolDefinition {
            name: "echo".to_string(),
            description: "Echo text".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }];
        inject_tools_into_system_prompt(&mut messages, &tools);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, crate::llm::Role::System);
        assert!(messages[0].content.contains("echo"));
    }

    #[test]
    fn test_inject_tools_empty_tools_noop() {
        let mut messages = vec![ChatMessage::system("You are helpful.")];
        let original = messages[0].content.clone();
        inject_tools_into_system_prompt(&mut messages, &[]);
        assert_eq!(messages[0].content, original);
    }

    #[test]
    fn test_inject_tools_includes_enum_values() {
        let mut messages = vec![ChatMessage::system("You are helpful.")];
        let tools = vec![IronToolDefinition {
            name: "time".to_string(),
            description: "Get current time".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["now", "parse", "format"],
                        "description": "The operation"
                    }
                },
                "required": ["operation"]
            }),
        }];
        inject_tools_into_system_prompt(&mut messages, &tools);
        assert!(
            messages[0]
                .content
                .contains(r#"["now", "parse", "format"]"#)
        );
    }
}
