//! LLM-facing tools for managing routines.
//!
//! Five tools let the agent manage routines conversationally:
//! - `routine_create` - Create a new routine
//! - `routine_list` - List all routines with status
//! - `routine_update` - Modify or toggle a routine
//! - `routine_delete` - Remove a routine
//! - `routine_history` - View past runs

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;

use crate::agent::routine::{NotifyConfig, Routine, RoutineAction, RoutineGuardrails, Trigger};
use crate::agent::routine_engine::RoutineEngine;
use crate::context::JobContext;
use crate::db::Database;
use crate::tools::tool::{Tool, ToolError, ToolOutput, require_str};

fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn truncate_for_error(input: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (idx, ch) in input.chars().enumerate() {
        if idx >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

fn summarize_json(value: &serde_json::Value) -> String {
    let rendered = value.to_string();
    truncate_for_error(&rendered, 160)
}

fn params_key_list(params: &serde_json::Value) -> String {
    if let Some(obj) = params.as_object() {
        let mut keys: Vec<&str> = obj.keys().map(std::string::String::as_str).collect();
        keys.sort_unstable();
        if keys.is_empty() {
            "<empty object>".to_string()
        } else {
            keys.join(", ")
        }
    } else {
        format!("<non-object: {}>", json_type_name(params))
    }
}

/// Accept either a cron string or a token array (e.g. ["0", "*/2", "*", "*", "*", "*"]).
fn parse_schedule_param(value: &serde_json::Value) -> Result<String, ToolError> {
    match value {
        serde_json::Value::String(s) => {
            if s.trim().is_empty() {
                return Err(ToolError::InvalidParameters(
                    "cron trigger requires non-empty 'schedule' string".to_string(),
                ));
            }
            Ok(s.clone())
        }
        serde_json::Value::Array(parts) => {
            if parts.is_empty() {
                return Err(ToolError::InvalidParameters(
                    "cron trigger requires non-empty 'schedule'; received []".to_string(),
                ));
            }

            let mut tokens = Vec::with_capacity(parts.len());
            for (idx, part) in parts.iter().enumerate() {
                match part {
                    serde_json::Value::String(s) => tokens.push(s.trim().to_string()),
                    serde_json::Value::Number(n) => tokens.push(n.to_string()),
                    other => {
                        return Err(ToolError::InvalidParameters(format!(
                            "cron schedule token at index {} must be string/number, got {} (value={})",
                            idx,
                            json_type_name(other),
                            summarize_json(other)
                        )));
                    }
                }
            }

            Ok(tokens.join(" "))
        }
        other => Err(ToolError::InvalidParameters(format!(
            "cron trigger requires 'schedule' as string or array of tokens; got {} (value={})",
            json_type_name(other),
            summarize_json(other)
        ))),
    }
}

fn parse_string_array_param(
    params: &serde_json::Value,
    key: &str,
) -> Result<Option<Vec<String>>, ToolError> {
    let Some(value) = params.get(key) else {
        return Ok(None);
    };

    let Some(items) = value.as_array() else {
        return Err(ToolError::InvalidParameters(format!(
            "'{}' must be an array of strings",
            key
        )));
    };

    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let Some(name) = item.as_str() else {
            return Err(ToolError::InvalidParameters(format!(
                "'{}' index {} must be a string",
                key, idx
            )));
        };
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(ToolError::InvalidParameters(format!(
                "'{}' index {} cannot be empty",
                key, idx
            )));
        }
        out.push(trimmed.to_string());
    }

    Ok(Some(out))
}

// ==================== routine_create ====================

pub struct RoutineCreateTool {
    store: Arc<dyn Database>,
    engine: Arc<RoutineEngine>,
}

impl RoutineCreateTool {
    pub fn new(store: Arc<dyn Database>, engine: Arc<RoutineEngine>) -> Self {
        Self { store, engine }
    }
}

#[async_trait]
impl Tool for RoutineCreateTool {
    fn name(&self) -> &str {
        "routine_create"
    }

    fn description(&self) -> &str {
        "Create a new routine (scheduled or event-driven task). \
         Supports cron schedules, event pattern matching, webhooks, and manual triggers. \
         Use this when the user wants something to happen periodically or reactively."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique name for the routine (e.g. 'daily-pr-review')"
                },
                "description": {
                    "type": "string",
                    "description": "What this routine does"
                },
                "trigger_type": {
                    "type": "string",
                    "enum": ["cron", "event", "webhook", "manual"],
                    "description": "When the routine fires"
                },
                "schedule": {
                    "type": "string",
                    "description": "Cron expression (for cron trigger). E.g. '0 9 * * MON-FRI' for weekdays at 9am. Uses 6-field cron (sec min hour day month weekday)."
                },
                "event_pattern": {
                    "type": "string",
                    "description": "Regex pattern to match messages (for event trigger)"
                },
                "event_channel": {
                    "type": "string",
                    "description": "Optional channel filter for event trigger (e.g. 'telegram')"
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt/instructions for the routine"
                },
                "context_paths": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Workspace paths to load as context (e.g. ['context/priorities.md'])"
                },
                "action_type": {
                    "type": "string",
                    "enum": ["lightweight", "full_job"],
                    "description": "Execution mode: 'lightweight' (single LLM call, default) or 'full_job' (multi-turn with tools)"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum iterations for full_job routines (default: 10)"
                },
                "tool_allowlist": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Per-job allowlist for approval-gated tools on full_job routines"
                },
                "tool_denylist": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Per-job denylist that overrides allowlist/global approvals on full_job routines"
                },
                "cooldown_secs": {
                    "type": "integer",
                    "description": "Minimum seconds between fires (default: 300)"
                }
            },
            "required": ["name", "trigger_type", "prompt"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let name = require_str(&params, "name")?;

        let description = params
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let trigger_type = require_str(&params, "trigger_type")?;

        let prompt = require_str(&params, "prompt")?;

        // Build trigger
        let trigger = match trigger_type {
            "cron" => {
                let Some(schedule_value) = params.get("schedule") else {
                    let keys = params_key_list(&params);
                    tracing::warn!(
                        tool = "routine_create",
                        trigger_type = "cron",
                        keys = %keys,
                        "Missing required cron schedule"
                    );
                    return Err(ToolError::InvalidParameters(format!(
                        "cron trigger requires 'schedule'. Received keys: [{}]. Example: \
                         schedule=\"0 */2 * * * *\" (6-field: sec min hour day month weekday)",
                        keys
                    )));
                };

                let schedule = parse_schedule_param(schedule_value).map_err(|e| {
                    tracing::warn!(
                        tool = "routine_create",
                        trigger_type = "cron",
                        schedule_raw = %summarize_json(schedule_value),
                        error = %e,
                        "Invalid cron schedule parameter"
                    );
                    e
                })?;
                // Validate cron expression
                self.engine.next_cron_fire(&schedule).map_err(|e| {
                    tracing::warn!(
                        tool = "routine_create",
                        trigger_type = "cron",
                        schedule = %schedule,
                        error = %e,
                        "Cron schedule validation failed"
                    );
                    ToolError::InvalidParameters(format!(
                        "invalid cron schedule: {e}. Received schedule={:?}. \
                         Expected 6-field cron: sec min hour day month weekday \
                         (example: \"0 */2 * * * *\")",
                        schedule
                    ))
                })?;
                Trigger::Cron { schedule }
            }
            "event" => {
                let pattern = params
                    .get("event_pattern")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        ToolError::InvalidParameters(
                            "event trigger requires 'event_pattern'".to_string(),
                        )
                    })?;
                // Validate regex
                regex::Regex::new(pattern)
                    .map_err(|e| ToolError::InvalidParameters(format!("invalid regex: {e}")))?;
                let channel = params
                    .get("event_channel")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                Trigger::Event {
                    channel,
                    pattern: pattern.to_string(),
                }
            }
            "webhook" => Trigger::Webhook {
                path: None,
                secret: None,
            },
            "manual" => Trigger::Manual,
            other => {
                return Err(ToolError::InvalidParameters(format!(
                    "unknown trigger_type: {other}"
                )));
            }
        };

        // Build action
        let action_type = params
            .get("action_type")
            .and_then(|v| v.as_str())
            .unwrap_or("lightweight");

        let context_paths: Vec<String> = params
            .get("context_paths")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let action = match action_type {
            "lightweight" => RoutineAction::Lightweight {
                prompt: prompt.to_string(),
                context_paths,
                max_tokens: 4096,
            },
            "full_job" => {
                let max_iterations = params
                    .get("max_iterations")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10) as u32;
                let tool_allowlist =
                    parse_string_array_param(&params, "tool_allowlist")?.unwrap_or_default();
                let tool_denylist =
                    parse_string_array_param(&params, "tool_denylist")?.unwrap_or_default();

                RoutineAction::FullJob {
                    title: name.to_string(),
                    description: prompt.to_string(),
                    max_iterations,
                    tool_allowlist,
                    tool_denylist,
                }
            }
            other => {
                return Err(ToolError::InvalidParameters(format!(
                    "unknown action_type: {other}"
                )));
            }
        };

        let cooldown_secs = params
            .get("cooldown_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(300);

        // Compute next fire time for cron
        let next_fire = if let Trigger::Cron { ref schedule } = trigger {
            self.engine.next_cron_fire(schedule).unwrap_or(None)
        } else {
            None
        };

        let routine = Routine {
            id: Uuid::new_v4(),
            name: name.to_string(),
            description: description.to_string(),
            user_id: ctx.user_id.clone(),
            enabled: true,
            trigger,
            action,
            guardrails: RoutineGuardrails {
                cooldown: Duration::from_secs(cooldown_secs),
                max_concurrent: 1,
                dedup_window: None,
            },
            notify: NotifyConfig::default(),
            last_run_at: None,
            next_fire_at: next_fire,
            run_count: 0,
            consecutive_failures: 0,
            state: serde_json::json!({}),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        self.store
            .create_routine(&routine)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to create routine: {e}")))?;

        // Refresh event cache if this is an event trigger
        if routine.trigger.type_tag() == "event" {
            self.engine.refresh_event_cache().await;
        }

        let result = serde_json::json!({
            "id": routine.id.to_string(),
            "name": routine.name,
            "trigger_type": routine.trigger.type_tag(),
            "next_fire_at": routine.next_fire_at.map(|t| t.to_rfc3339()),
            "status": "created",
        });

        Ok(ToolOutput::success(result, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false
    }
}

// ==================== routine_list ====================

pub struct RoutineListTool {
    store: Arc<dyn Database>,
}

impl RoutineListTool {
    pub fn new(store: Arc<dyn Database>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for RoutineListTool {
    fn name(&self) -> &str {
        "routine_list"
    }

    fn description(&self) -> &str {
        "List all routines with their status, trigger info, and next fire time."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(
        &self,
        _params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let routines = self
            .store
            .list_routines(&ctx.user_id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to list routines: {e}")))?;

        let list: Vec<serde_json::Value> = routines
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id.to_string(),
                    "name": r.name,
                    "description": r.description,
                    "enabled": r.enabled,
                    "trigger_type": r.trigger.type_tag(),
                    "action_type": r.action.type_tag(),
                    "last_run_at": r.last_run_at.map(|t| t.to_rfc3339()),
                    "next_fire_at": r.next_fire_at.map(|t| t.to_rfc3339()),
                    "run_count": r.run_count,
                    "consecutive_failures": r.consecutive_failures,
                })
            })
            .collect();

        let result = serde_json::json!({
            "count": list.len(),
            "routines": list,
        });

        Ok(ToolOutput::success(result, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false
    }
}

// ==================== routine_update ====================

pub struct RoutineUpdateTool {
    store: Arc<dyn Database>,
    engine: Arc<RoutineEngine>,
}

impl RoutineUpdateTool {
    pub fn new(store: Arc<dyn Database>, engine: Arc<RoutineEngine>) -> Self {
        Self { store, engine }
    }
}

#[async_trait]
impl Tool for RoutineUpdateTool {
    fn name(&self) -> &str {
        "routine_update"
    }

    fn description(&self) -> &str {
        "Update an existing routine. Can modify trigger, prompt, schedule, or toggle enabled state. \
         Pass the routine name and only the fields you want to change."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the routine to update"
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Enable or disable the routine"
                },
                "prompt": {
                    "type": "string",
                    "description": "New prompt/instructions"
                },
                "schedule": {
                    "type": "string",
                    "description": "New cron schedule (for cron triggers)"
                },
                "description": {
                    "type": "string",
                    "description": "New description"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "New max iterations for full_job routines"
                },
                "tool_allowlist": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "New per-job allowlist for full_job routines"
                },
                "tool_denylist": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "New per-job denylist for full_job routines"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let name = require_str(&params, "name")?;

        let mut routine = self
            .store
            .get_routine_by_name(&ctx.user_id, name)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("DB error: {e}")))?
            .ok_or_else(|| ToolError::ExecutionFailed(format!("routine '{}' not found", name)))?;

        // Apply updates
        if let Some(enabled) = params.get("enabled").and_then(|v| v.as_bool()) {
            routine.enabled = enabled;
        }

        if let Some(desc) = params.get("description").and_then(|v| v.as_str()) {
            routine.description = desc.to_string();
        }

        if let Some(prompt) = params.get("prompt").and_then(|v| v.as_str()) {
            match &mut routine.action {
                RoutineAction::Lightweight { prompt: p, .. } => *p = prompt.to_string(),
                RoutineAction::FullJob { description: d, .. } => *d = prompt.to_string(),
            }
        }

        if let Some(max_iterations) = params.get("max_iterations").and_then(|v| v.as_u64())
            && let RoutineAction::FullJob {
                max_iterations: current,
                ..
            } = &mut routine.action
        {
            *current = max_iterations as u32;
        }

        if let Some(tool_allowlist) = parse_string_array_param(&params, "tool_allowlist")?
            && let RoutineAction::FullJob {
                tool_allowlist: current,
                ..
            } = &mut routine.action
        {
            *current = tool_allowlist;
        }

        if let Some(tool_denylist) = parse_string_array_param(&params, "tool_denylist")?
            && let RoutineAction::FullJob {
                tool_denylist: current,
                ..
            } = &mut routine.action
        {
            *current = tool_denylist;
        }

        if let Some(schedule) = params.get("schedule") {
            let schedule = parse_schedule_param(schedule).map_err(|e| {
                tracing::warn!(
                    tool = "routine_update",
                    schedule_raw = %summarize_json(schedule),
                    error = %e,
                    "Invalid routine_update schedule parameter"
                );
                e
            })?;
            // Validate
            self.engine.next_cron_fire(&schedule).map_err(|e| {
                tracing::warn!(
                    tool = "routine_update",
                    schedule = %schedule,
                    error = %e,
                    "routine_update cron validation failed"
                );
                ToolError::InvalidParameters(format!(
                    "invalid cron schedule for routine_update: {e}. Received schedule={:?}. \
                         Expected 6-field cron: sec min hour day month weekday \
                         (example: \"0 */2 * * * *\")",
                    schedule
                ))
            })?;

            routine.trigger = Trigger::Cron {
                schedule: schedule.clone(),
            };
            routine.next_fire_at = self.engine.next_cron_fire(&schedule).unwrap_or(None);
        }

        self.store
            .update_routine(&routine)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to update: {e}")))?;

        // Refresh event cache in case trigger changed
        self.engine.refresh_event_cache().await;

        let result = serde_json::json!({
            "name": routine.name,
            "enabled": routine.enabled,
            "trigger_type": routine.trigger.type_tag(),
            "next_fire_at": routine.next_fire_at.map(|t| t.to_rfc3339()),
            "status": "updated",
        });

        Ok(ToolOutput::success(result, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false
    }
}

// ==================== routine_delete ====================

pub struct RoutineDeleteTool {
    store: Arc<dyn Database>,
    engine: Arc<RoutineEngine>,
}

impl RoutineDeleteTool {
    pub fn new(store: Arc<dyn Database>, engine: Arc<RoutineEngine>) -> Self {
        Self { store, engine }
    }
}

#[async_trait]
impl Tool for RoutineDeleteTool {
    fn name(&self) -> &str {
        "routine_delete"
    }

    fn description(&self) -> &str {
        "Delete a routine permanently. This also removes all run history."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the routine to delete"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let name = require_str(&params, "name")?;

        let routine = self
            .store
            .get_routine_by_name(&ctx.user_id, name)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("DB error: {e}")))?
            .ok_or_else(|| ToolError::ExecutionFailed(format!("routine '{}' not found", name)))?;

        let deleted = self
            .store
            .delete_routine(routine.id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to delete: {e}")))?;

        // Refresh event cache
        self.engine.refresh_event_cache().await;

        let result = serde_json::json!({
            "name": name,
            "deleted": deleted,
        });

        Ok(ToolOutput::success(result, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false
    }
}

// ==================== routine_history ====================

pub struct RoutineHistoryTool {
    store: Arc<dyn Database>,
}

impl RoutineHistoryTool {
    pub fn new(store: Arc<dyn Database>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for RoutineHistoryTool {
    fn name(&self) -> &str {
        "routine_history"
    }

    fn description(&self) -> &str {
        "View the execution history of a routine. Shows recent runs with status, duration, and results."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the routine"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max runs to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(
        &self,
        params: serde_json::Value,
        ctx: &JobContext,
    ) -> Result<ToolOutput, ToolError> {
        let start = std::time::Instant::now();

        let name = require_str(&params, "name")?;

        let limit = params
            .get("limit")
            .and_then(|v| v.as_i64())
            .unwrap_or(10)
            .min(50);

        let routine = self
            .store
            .get_routine_by_name(&ctx.user_id, name)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("DB error: {e}")))?
            .ok_or_else(|| ToolError::ExecutionFailed(format!("routine '{}' not found", name)))?;

        let runs = self
            .store
            .list_routine_runs(routine.id, limit)
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("failed to list runs: {e}")))?;

        let run_list: Vec<serde_json::Value> = runs
            .iter()
            .map(|r| {
                let duration_secs = r
                    .completed_at
                    .map(|c| c.signed_duration_since(r.started_at).num_seconds());
                serde_json::json!({
                    "id": r.id.to_string(),
                    "trigger_type": r.trigger_type,
                    "trigger_detail": r.trigger_detail,
                    "started_at": r.started_at.to_rfc3339(),
                    "completed_at": r.completed_at.map(|t| t.to_rfc3339()),
                    "duration_secs": duration_secs,
                    "status": r.status.to_string(),
                    "result_summary": r.result_summary,
                    "tokens_used": r.tokens_used,
                })
            })
            .collect();

        let result = serde_json::json!({
            "routine": name,
            "total_runs": routine.run_count,
            "runs": run_list,
        });

        Ok(ToolOutput::success(result, start.elapsed()))
    }

    fn requires_sanitization(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_schedule_accepts_string() {
        let value = serde_json::json!("0 */2 * * * *");
        let parsed = crate::tools::builtin::routine::parse_schedule_param(&value)
            .expect("string schedule should parse");
        assert_eq!(parsed, "0 */2 * * * *");
    }

    #[test]
    fn parse_schedule_accepts_token_array() {
        let value = serde_json::json!(["0", "*/2", "*", "*", "*", "*"]);
        let parsed = crate::tools::builtin::routine::parse_schedule_param(&value)
            .expect("array schedule should parse");
        assert_eq!(parsed, "0 */2 * * * *");
    }

    #[test]
    fn parse_schedule_rejects_invalid_array_item_type() {
        let value = serde_json::json!(["0", {"bad": "token"}]);
        let err = crate::tools::builtin::routine::parse_schedule_param(&value)
            .expect_err("object token should fail");
        assert!(err.to_string().contains("must be string/number"));
    }
}
