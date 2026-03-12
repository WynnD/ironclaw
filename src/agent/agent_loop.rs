//! Main agent loop.
//!
//! Contains the `Agent` struct, `AgentDeps`, and the core event loop (`run`).
//! The heavy lifting is delegated to sibling modules:
//!
//! - `dispatcher` - Tool dispatch (agentic loop, tool execution)
//! - `commands` - System commands and job handlers
//! - `thread_ops` - Thread/session operations (user input, undo, approval, persistence)

use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::agent::context_monitor::{ContextMonitor, estimate_text_tokens};
use crate::agent::heartbeat::spawn_heartbeat;
use crate::agent::routine_engine::{RoutineEngine, spawn_cron_ticker};
use crate::agent::self_repair::{DefaultSelfRepair, RepairResult, SelfRepair};
use crate::agent::session::ThreadState;
use crate::agent::session_manager::SessionManager;
use crate::agent::submission::{Submission, SubmissionParser, SubmissionResult};
use crate::agent::{HeartbeatConfig as AgentHeartbeatConfig, Router, Scheduler};
use crate::channels::{ChannelManager, IncomingMessage, OutgoingResponse, StatusUpdate};
use crate::config::{AgentConfig, HeartbeatConfig, RoutineConfig, SkillsConfig};
use crate::context::ContextManager;
use crate::db::Database;
use crate::error::Error;
use crate::extensions::ExtensionManager;
use crate::hooks::HookRegistry;
use crate::llm::LlmProvider;
use crate::safety::SafetyLayer;
use crate::skills::SkillRegistry;
use crate::tools::ToolRegistry;
use crate::workspace::Workspace;

/// Collapse a tool output string into a single-line preview for display.
pub(crate) fn truncate_for_preview(output: &str, max_chars: usize) -> String {
    let collapsed: String = output
        .chars()
        .take(max_chars + 50)
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    // char_indices gives us byte offsets at char boundaries, so the slice is always valid UTF-8.
    if collapsed.chars().count() > max_chars {
        let byte_offset = collapsed
            .char_indices()
            .nth(max_chars)
            .map(|(i, _)| i)
            .unwrap_or(collapsed.len());
        format!("{}...", &collapsed[..byte_offset])
    } else {
        collapsed
    }
}

/// Render tool parameters as a very short one-line JSON preview for status UIs.
pub(crate) fn tool_params_preview(params: &serde_json::Value, max_chars: usize) -> String {
    let serialized = serde_json::to_string(params).unwrap_or_else(|_| params.to_string());
    truncate_for_preview(&serialized, max_chars)
}

/// Convert external channel thread IDs into a stable internal UUID string.
///
/// - If `external_thread_id` is already a UUID, preserve it.
/// - Otherwise derive a deterministic UUID from channel + user + external ID.
///
/// This lets channels like Telegram (numeric chat IDs) map to the same
/// internal conversation across process restarts.
fn canonicalize_external_thread_id(
    channel: &str,
    user_id: &str,
    external_thread_id: &str,
) -> String {
    if Uuid::parse_str(external_thread_id).is_ok() {
        return external_thread_id.to_string();
    }

    let key = format!("{channel}:{user_id}:{external_thread_id}");
    Uuid::new_v5(&Uuid::NAMESPACE_URL, key.as_bytes()).to_string()
}

fn normalize_approval_submission_for_thread(
    submission: Submission,
    original_content: &str,
    thread_state: Option<ThreadState>,
) -> Result<Submission, String> {
    if matches!(submission, Submission::ApprovalResponse { .. })
        && thread_state != Some(ThreadState::AwaitingApproval)
    {
        if original_content.trim_start().starts_with('/') {
            return Err("Error: No pending approval request.".to_string());
        }
        return Ok(Submission::UserInput {
            content: original_content.to_string(),
        });
    }

    Ok(submission)
}

/// Core dependencies for the agent.
///
/// Bundles the shared components to reduce argument count.
pub struct AgentDeps {
    pub store: Option<Arc<dyn Database>>,
    pub llm: Arc<dyn LlmProvider>,
    /// Cheap/fast LLM for lightweight tasks (heartbeat, routing, evaluation).
    /// Falls back to the main `llm` if None.
    pub cheap_llm: Option<Arc<dyn LlmProvider>>,
    pub safety: Arc<SafetyLayer>,
    pub tools: Arc<ToolRegistry>,
    pub workspace: Option<Arc<Workspace>>,
    pub extension_manager: Option<Arc<ExtensionManager>>,
    pub skill_registry: Option<Arc<std::sync::RwLock<SkillRegistry>>>,
    pub skill_catalog: Option<Arc<crate::skills::catalog::SkillCatalog>>,
    pub skills_config: SkillsConfig,
    pub hooks: Arc<HookRegistry>,
    /// Cost enforcement guardrails (daily budget, hourly rate limits).
    pub cost_guard: Arc<crate::agent::cost_guard::CostGuard>,
    /// SSE broadcast sender for live job event streaming to the web gateway.
    pub sse_tx: Option<tokio::sync::broadcast::Sender<crate::channels::web::types::SseEvent>>,
}

/// The main agent that coordinates all components.
pub struct Agent {
    pub(super) config: AgentConfig,
    pub(super) deps: AgentDeps,
    pub(super) channels: Arc<ChannelManager>,
    pub(super) context_manager: Arc<ContextManager>,
    pub(super) scheduler: Arc<Scheduler>,
    pub(super) router: Router,
    pub(super) session_manager: Arc<SessionManager>,
    pub(super) context_monitor: Arc<RwLock<ContextMonitor>>,
    pub(super) heartbeat_config: Option<HeartbeatConfig>,
    pub(super) hygiene_config: Option<crate::config::HygieneConfig>,
    pub(super) routine_config: Option<RoutineConfig>,
}

impl Agent {
    /// Create a new agent.
    ///
    /// Optionally accepts pre-created `ContextManager` and `SessionManager` for sharing
    /// with external components (job tools, web gateway). Creates new ones if not provided.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: AgentConfig,
        deps: AgentDeps,
        channels: Arc<ChannelManager>,
        heartbeat_config: Option<HeartbeatConfig>,
        hygiene_config: Option<crate::config::HygieneConfig>,
        routine_config: Option<RoutineConfig>,
        context_manager: Option<Arc<ContextManager>>,
        session_manager: Option<Arc<SessionManager>>,
    ) -> Self {
        let context_manager = context_manager
            .unwrap_or_else(|| Arc::new(ContextManager::new(config.max_parallel_jobs)));

        let session_manager = session_manager.unwrap_or_else(|| Arc::new(SessionManager::new()));

        let mut scheduler = Scheduler::new(
            config.clone(),
            context_manager.clone(),
            deps.llm.clone(),
            deps.safety.clone(),
            deps.tools.clone(),
            deps.store.clone(),
            deps.hooks.clone(),
        );
        if let Some(ref tx) = deps.sse_tx {
            scheduler.set_sse_sender(tx.clone());
        }
        let scheduler = Arc::new(scheduler);

        Self {
            config,
            deps,
            channels,
            context_manager,
            scheduler,
            router: Router::new(),
            session_manager,
            context_monitor: Arc::new(RwLock::new(ContextMonitor::new())),
            heartbeat_config,
            hygiene_config,
            routine_config,
        }
    }

    // Convenience accessors

    /// Get the scheduler (for external wiring, e.g. CreateJobTool).
    pub fn scheduler(&self) -> Arc<Scheduler> {
        Arc::clone(&self.scheduler)
    }

    /// Get the context monitor (for external wiring, e.g. web gateway live settings).
    pub fn context_monitor(&self) -> Arc<RwLock<ContextMonitor>> {
        Arc::clone(&self.context_monitor)
    }

    pub(super) fn store(&self) -> Option<&Arc<dyn Database>> {
        self.deps.store.as_ref()
    }

    pub(super) fn llm(&self) -> &Arc<dyn LlmProvider> {
        &self.deps.llm
    }

    /// Get the cheap/fast LLM provider, falling back to the main one.
    pub(super) fn cheap_llm(&self) -> &Arc<dyn LlmProvider> {
        self.deps.cheap_llm.as_ref().unwrap_or(&self.deps.llm)
    }

    pub(super) fn safety(&self) -> &Arc<SafetyLayer> {
        &self.deps.safety
    }

    pub(super) fn tools(&self) -> &Arc<ToolRegistry> {
        &self.deps.tools
    }

    pub(super) fn workspace(&self) -> Option<&Arc<Workspace>> {
        self.deps.workspace.as_ref()
    }

    pub(super) fn hooks(&self) -> &Arc<HookRegistry> {
        &self.deps.hooks
    }

    pub(super) fn cost_guard(&self) -> &Arc<crate::agent::cost_guard::CostGuard> {
        &self.deps.cost_guard
    }

    /// Refresh the base context limit from the active model metadata (unless overridden).
    ///
    /// Priority: env `CONTEXT_LIMIT` > DB `context_limit` setting > model metadata > fallback.
    pub(super) async fn refresh_context_monitor_limit(&self) -> usize {
        let base_limit = if let Some(limit) = self.config.context_limit_override {
            limit
        } else {
            // Check DB-persisted context_limit (set via web UI settings).
            let user_id = std::env::var("GATEWAY_USER_ID").unwrap_or_else(|_| "default".to_string());
            let db_limit = if let Some(ref store) = self.deps.store {
                store
                    .get_setting(&user_id, "context_limit")
                    .await
                    .ok()
                    .flatten()
                    .and_then(|v| v.as_u64().map(|n| n as usize))
            } else {
                None
            };

            if let Some(limit) = db_limit {
                tracing::debug!(context_limit = limit, "Using DB-persisted context limit");
                limit
            } else {
                let fallback = {
                    let monitor = self.context_monitor.read().await;
                    monitor.limit()
                };
                match self.llm().model_metadata().await {
                    Ok(metadata) => {
                        let limit = metadata
                            .context_length
                            .map(|v| v as usize)
                            .unwrap_or(fallback);
                        tracing::debug!(
                            model = %metadata.id,
                            context_limit = limit,
                            "Resolved model context limit"
                        );
                        limit
                    }
                    Err(err) => {
                        tracing::warn!(
                            "Failed to fetch model metadata for context limit: {}",
                            err
                        );
                        fallback
                    }
                }
            }
        };

        let mut monitor = self.context_monitor.write().await;
        *monitor = monitor.clone().with_limit(base_limit);
        base_limit
    }

    /// Clone the current monitor and apply a reserved-token headroom.
    pub(super) async fn context_monitor_with_reserved(
        &self,
        reserved_tokens: usize,
    ) -> ContextMonitor {
        let monitor = self.context_monitor.read().await.clone();
        let effective_limit = monitor.limit().saturating_sub(reserved_tokens).max(1024);
        monitor.with_limit(effective_limit)
    }

    /// Sanitize, cap, and wrap tool output before adding it to LLM context.
    pub(super) fn format_tool_result_for_context(&self, tool_name: &str, output: &str) -> String {
        let sanitized = self.safety().sanitize_tool_output(tool_name, output);
        let capped_content = self
            .cap_text_for_context_tokens(&sanitized.content, self.config.max_tool_output_tokens);
        self.safety()
            .wrap_for_llm(tool_name, &capped_content, sanitized.was_modified)
    }

    fn cap_text_for_context_tokens(&self, content: &str, max_tokens: usize) -> String {
        if max_tokens == 0 {
            return content.to_string();
        }

        let estimated_tokens = estimate_text_tokens(content);
        if estimated_tokens <= max_tokens {
            return content.to_string();
        }

        let suffix = format!(
            "\n\n[tool output truncated for context: estimated {} tokens exceeds limit {}]",
            estimated_tokens, max_tokens
        );
        let suffix_tokens = estimate_text_tokens(&suffix);
        if suffix_tokens >= max_tokens {
            return suffix;
        }

        let target_tokens = max_tokens.saturating_sub(suffix_tokens);
        let total_chars = content.chars().count();
        let estimated_chars = ((total_chars as f64)
            * (target_tokens as f64 / estimated_tokens as f64))
            .floor() as usize;
        let min_keep = total_chars.min(64);
        let keep_chars = estimated_chars.clamp(min_keep, total_chars);

        let byte_end = content
            .char_indices()
            .nth(keep_chars)
            .map(|(idx, _)| idx)
            .unwrap_or(content.len());

        let mut truncated = content[..byte_end].to_string();
        truncated.push_str(&suffix);
        truncated
    }

    pub(super) fn skill_registry(&self) -> Option<&Arc<std::sync::RwLock<SkillRegistry>>> {
        self.deps.skill_registry.as_ref()
    }

    pub(super) fn skill_catalog(&self) -> Option<&Arc<crate::skills::catalog::SkillCatalog>> {
        self.deps.skill_catalog.as_ref()
    }

    /// Select active skills for a message using deterministic prefiltering.
    pub(super) fn select_active_skills(
        &self,
        message_content: &str,
    ) -> Vec<crate::skills::LoadedSkill> {
        if let Some(registry) = self.skill_registry() {
            let guard = match registry.read() {
                Ok(g) => g,
                Err(e) => {
                    tracing::error!("Skill registry lock poisoned: {}", e);
                    return vec![];
                }
            };
            let available = guard.skills();
            let skills_cfg = &self.deps.skills_config;
            let selected = crate::skills::prefilter_skills(
                message_content,
                available,
                skills_cfg.max_active_skills,
                skills_cfg.max_context_tokens,
            );

            if !selected.is_empty() {
                tracing::debug!(
                    "Selected {} skill(s) for message: {}",
                    selected.len(),
                    selected
                        .iter()
                        .map(|s| s.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            selected.into_iter().cloned().collect()
        } else {
            vec![]
        }
    }

    /// Run the agent main loop.
    pub async fn run(self) -> Result<(), Error> {
        let _ = self.refresh_context_monitor_limit().await;

        // Start channels
        let mut message_stream = self.channels.start_all().await?;

        // Start self-repair task with notification forwarding
        let repair = Arc::new(DefaultSelfRepair::new(
            self.context_manager.clone(),
            self.config.stuck_threshold,
            self.config.max_repair_attempts,
        ));
        let repair_interval = self.config.repair_check_interval;
        let repair_channels = self.channels.clone();
        let repair_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(repair_interval).await;

                // Check stuck jobs
                let stuck_jobs = repair.detect_stuck_jobs().await;
                for job in stuck_jobs {
                    tracing::info!("Attempting to repair stuck job {}", job.job_id);
                    let result = repair.repair_stuck_job(&job).await;
                    let notification = match &result {
                        Ok(RepairResult::Success { message }) => {
                            tracing::info!("Repair succeeded: {}", message);
                            Some(format!(
                                "Job {} was stuck for {}s, recovery succeeded: {}",
                                job.job_id,
                                job.stuck_duration.as_secs(),
                                message
                            ))
                        }
                        Ok(RepairResult::Failed { message }) => {
                            tracing::error!("Repair failed: {}", message);
                            Some(format!(
                                "Job {} was stuck for {}s, recovery failed permanently: {}",
                                job.job_id,
                                job.stuck_duration.as_secs(),
                                message
                            ))
                        }
                        Ok(RepairResult::ManualRequired { message }) => {
                            tracing::warn!("Manual intervention needed: {}", message);
                            Some(format!(
                                "Job {} needs manual intervention: {}",
                                job.job_id, message
                            ))
                        }
                        Ok(RepairResult::Retry { message }) => {
                            tracing::warn!("Repair needs retry: {}", message);
                            None // Don't spam the user on retries
                        }
                        Err(e) => {
                            tracing::error!("Repair error: {}", e);
                            None
                        }
                    };

                    if let Some(msg) = notification {
                        let response = OutgoingResponse::text(format!("Self-Repair: {}", msg));
                        let _ = repair_channels.broadcast_all("default", response).await;
                    }
                }

                // Check broken tools
                let broken_tools = repair.detect_broken_tools().await;
                for tool in broken_tools {
                    tracing::info!("Attempting to repair broken tool: {}", tool.name);
                    match repair.repair_broken_tool(&tool).await {
                        Ok(RepairResult::Success { message }) => {
                            let response = OutgoingResponse::text(format!(
                                "Self-Repair: Tool '{}' repaired: {}",
                                tool.name, message
                            ));
                            let _ = repair_channels.broadcast_all("default", response).await;
                        }
                        Ok(result) => {
                            tracing::info!("Tool repair result: {:?}", result);
                        }
                        Err(e) => {
                            tracing::error!("Tool repair error: {}", e);
                        }
                    }
                }
            }
        });

        // Spawn session pruning task
        let session_mgr = self.session_manager.clone();
        let session_idle_timeout = self.config.session_idle_timeout;
        let pruning_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(600)); // Every 10 min
            interval.tick().await; // Skip immediate first tick
            loop {
                interval.tick().await;
                session_mgr.prune_stale_sessions(session_idle_timeout).await;
            }
        });

        // Spawn heartbeat if enabled
        let heartbeat_handle = if let Some(ref hb_config) = self.heartbeat_config {
            if hb_config.enabled {
                if let Some(workspace) = self.workspace() {
                    let config = AgentHeartbeatConfig::default()
                        .with_interval(std::time::Duration::from_secs(hb_config.interval_secs));

                    // Set up notification channel
                    let (notify_tx, mut notify_rx) =
                        tokio::sync::mpsc::channel::<OutgoingResponse>(16);

                    // Spawn notification forwarder that routes through channel manager
                    let notify_channel = hb_config.notify_channel.clone();
                    let notify_user = hb_config.notify_user.clone();
                    let channels = self.channels.clone();
                    tokio::spawn(async move {
                        while let Some(response) = notify_rx.recv().await {
                            let user = notify_user.as_deref().unwrap_or("default");

                            // Try the configured channel first, fall back to
                            // broadcasting on all channels.
                            let targeted_ok = if let Some(ref channel) = notify_channel {
                                channels
                                    .broadcast(channel, user, response.clone())
                                    .await
                                    .is_ok()
                            } else {
                                false
                            };

                            if !targeted_ok {
                                let results = channels.broadcast_all(user, response).await;
                                for (ch, result) in results {
                                    if let Err(e) = result {
                                        tracing::warn!(
                                            "Failed to broadcast heartbeat to {}: {}",
                                            ch,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    });

                    let hygiene = self
                        .hygiene_config
                        .as_ref()
                        .map(|h| h.to_workspace_config())
                        .unwrap_or_default();

                    Some(spawn_heartbeat(
                        config,
                        hygiene,
                        workspace.clone(),
                        self.cheap_llm().clone(),
                        self.safety().clone(),
                        Some(notify_tx),
                    ))
                } else {
                    tracing::warn!("Heartbeat enabled but no workspace available");
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Spawn routine engine if enabled
        let routine_handle = if let Some(ref rt_config) = self.routine_config {
            if rt_config.enabled {
                if let (Some(store), Some(workspace)) = (self.store(), self.workspace()) {
                    // Set up notification channel (same pattern as heartbeat)
                    let (notify_tx, mut notify_rx) =
                        tokio::sync::mpsc::channel::<OutgoingResponse>(32);

                    let engine = Arc::new(RoutineEngine::new(
                        rt_config.clone(),
                        Arc::clone(store),
                        self.llm().clone(),
                        Arc::clone(workspace),
                        notify_tx,
                        Some(self.scheduler.clone()),
                    ));

                    // Register routine tools
                    self.deps
                        .tools
                        .register_routine_tools(Arc::clone(store), Arc::clone(&engine));

                    // Load initial event cache
                    engine.refresh_event_cache().await;

                    // Spawn notification forwarder (mirrors heartbeat pattern)
                    let channels = self.channels.clone();
                    tokio::spawn(async move {
                        while let Some(response) = notify_rx.recv().await {
                            let user = response
                                .metadata
                                .get("notify_user")
                                .and_then(|v| v.as_str())
                                .unwrap_or("default")
                                .to_string();
                            let notify_channel = response
                                .metadata
                                .get("notify_channel")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());

                            // Try the configured channel first, fall back to
                            // broadcasting on all channels.
                            tracing::info!(
                                notify_channel = ?notify_channel,
                                user = %user,
                                "Dispatching routine notification"
                            );

                            let targeted_ok = if let Some(ref channel) = notify_channel {
                                match channels.broadcast(channel, &user, response.clone()).await {
                                    Ok(()) => true,
                                    Err(e) => {
                                        tracing::warn!(
                                            channel = %channel,
                                            user = %user,
                                            "Targeted routine notification failed: {}",
                                            e
                                        );
                                        false
                                    }
                                }
                            } else {
                                false
                            };

                            if !targeted_ok {
                                let results = channels.broadcast_all(&user, response).await;
                                for (ch, result) in results {
                                    if let Err(e) = result {
                                        tracing::warn!(
                                            "Failed to broadcast routine notification to {}: {}",
                                            ch,
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    });

                    // Spawn cron ticker
                    let cron_interval =
                        std::time::Duration::from_secs(rt_config.cron_check_interval_secs);
                    let cron_handle = spawn_cron_ticker(Arc::clone(&engine), cron_interval);

                    // Store engine reference for event trigger checking
                    // Safety: we're in run() which takes self, no other reference exists
                    let engine_ref = Arc::clone(&engine);
                    // SAFETY: self is consumed by run(), we can smuggle the engine in
                    // via a local to use in the message loop below.

                    tracing::info!(
                        "Routines enabled: cron ticker every {}s, max {} concurrent, cron timezone {}",
                        rt_config.cron_check_interval_secs,
                        rt_config.max_concurrent_routines,
                        rt_config.cron_timezone
                    );

                    Some((cron_handle, engine_ref))
                } else {
                    tracing::warn!("Routines enabled but store/workspace not available");
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Extract engine ref for use in message loop
        let routine_engine_for_loop = routine_handle.as_ref().map(|(_, e)| Arc::clone(e));

        // Main message loop
        tracing::info!("Agent {} ready and listening", self.config.name);

        loop {
            let message = tokio::select! {
                biased;
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Ctrl+C received, shutting down...");
                    break;
                }
                msg = message_stream.next() => {
                    match msg {
                        Some(m) => m,
                        None => {
                            tracing::info!("All channel streams ended, shutting down...");
                            break;
                        }
                    }
                }
            };

            match self.handle_message(&message).await {
                Ok(Some(response)) if !response.is_empty() => {
                    // Hook: BeforeOutbound — allow hooks to modify or suppress outbound
                    let event = crate::hooks::HookEvent::Outbound {
                        user_id: message.user_id.clone(),
                        channel: message.channel.clone(),
                        content: response.clone(),
                        thread_id: message.thread_id.clone(),
                    };
                    match self.hooks().run(&event).await {
                        Err(err) => {
                            tracing::warn!("BeforeOutbound hook blocked response: {}", err);
                        }
                        Ok(crate::hooks::HookOutcome::Continue {
                            modified: Some(new_content),
                        }) => {
                            if let Err(e) = self
                                .channels
                                .respond(&message, OutgoingResponse::text(new_content))
                                .await
                            {
                                tracing::error!(
                                    channel = %message.channel,
                                    error = %e,
                                    "Failed to send response to channel"
                                );
                            }
                        }
                        _ => {
                            if let Err(e) = self
                                .channels
                                .respond(&message, OutgoingResponse::text(response))
                                .await
                            {
                                tracing::error!(
                                    channel = %message.channel,
                                    error = %e,
                                    "Failed to send response to channel"
                                );
                            }
                        }
                    }
                }
                Ok(Some(empty)) => {
                    // Empty response, nothing to send (e.g. approval handled via send_status)
                    tracing::debug!(
                        channel = %message.channel,
                        user = %message.user_id,
                        empty_len = empty.len(),
                        "Suppressed empty response (not sent to channel)"
                    );
                }
                Ok(None) => {
                    // Shutdown signal received (/quit, /exit, /shutdown)
                    tracing::info!("Shutdown command received, exiting...");
                    break;
                }
                Err(e) => {
                    tracing::error!("Error handling message: {}", e);
                    if let Err(send_err) = self
                        .channels
                        .respond(&message, OutgoingResponse::text(format!("Error: {}", e)))
                        .await
                    {
                        tracing::error!(
                            channel = %message.channel,
                            error = %send_err,
                            "Failed to send error response to channel"
                        );
                    }
                }
            }

            // Check event triggers (cheap in-memory regex, fires async if matched)
            if let Some(ref engine) = routine_engine_for_loop {
                let fired = engine.check_event_triggers(&message).await;
                if fired > 0 {
                    tracing::debug!("Fired {} event-triggered routines", fired);
                }
            }
        }

        // Cleanup
        tracing::info!("Agent shutting down...");
        repair_handle.abort();
        pruning_handle.abort();
        if let Some(handle) = heartbeat_handle {
            handle.abort();
        }
        if let Some((cron_handle, _)) = routine_handle {
            cron_handle.abort();
        }
        self.scheduler.stop_all().await;
        self.channels.shutdown_all().await?;

        Ok(())
    }

    async fn handle_message(&self, message: &IncomingMessage) -> Result<Option<String>, Error> {
        // Parse submission type first
        let mut submission = SubmissionParser::parse(&message.content);

        // Hook: BeforeInbound — allow hooks to modify or reject user input
        if let Submission::UserInput { ref content } = submission {
            let event = crate::hooks::HookEvent::Inbound {
                user_id: message.user_id.clone(),
                channel: message.channel.clone(),
                content: content.clone(),
                thread_id: message.thread_id.clone(),
            };
            match self.hooks().run(&event).await {
                Err(crate::hooks::HookError::Rejected { reason }) => {
                    return Ok(Some(format!("[Message rejected: {}]", reason)));
                }
                Err(err) => {
                    return Ok(Some(format!("[Message blocked by hook policy: {}]", err)));
                }
                Ok(crate::hooks::HookOutcome::Continue {
                    modified: Some(new_content),
                }) => {
                    submission = Submission::UserInput {
                        content: new_content,
                    };
                }
                _ => {} // Continue, fail-open errors already logged in registry
            }
        }

        // Normalize external thread IDs to stable UUIDs so non-UUID channels
        // (e.g. Telegram chat IDs) can restore context after restart.
        let normalized_external_thread_id = message
            .thread_id
            .as_deref()
            .map(|tid| canonicalize_external_thread_id(&message.channel, &message.user_id, tid));

        // Hydrate thread from DB if it's a historical thread not in memory
        if let Some(ref external_thread_id) = normalized_external_thread_id {
            self.maybe_hydrate_thread(message, external_thread_id).await;
        }

        // Resolve session and thread
        let (session, thread_id) = self
            .session_manager
            .resolve_thread(
                &message.user_id,
                &message.channel,
                normalized_external_thread_id.as_deref(),
            )
            .await;

        // Approval aliases like "yes"/"ok" should only be treated as approval
        // responses when this thread is actually awaiting approval. Otherwise
        // they are normal user messages and must not get dropped.
        if matches!(submission, Submission::ApprovalResponse { .. }) {
            let thread_state = {
                let sess = session.lock().await;
                sess.threads.get(&thread_id).map(|t| t.state)
            };

            submission = match normalize_approval_submission_for_thread(
                submission,
                &message.content,
                thread_state,
            ) {
                Ok(s) => s,
                Err(msg) => return Ok(Some(msg)),
            };
        }

        // Auth mode interception: if the thread is awaiting a token, route
        // the message directly to the credential store. Nothing touches
        // logs, turns, history, or compaction.
        let pending_auth = {
            let sess = session.lock().await;
            sess.threads
                .get(&thread_id)
                .and_then(|t| t.pending_auth.clone())
        };

        if let Some(pending) = pending_auth {
            match &submission {
                Submission::UserInput { content } => {
                    return self
                        .process_auth_token(message, &pending, content, session, thread_id)
                        .await;
                }
                _ => {
                    // Any control submission (interrupt, undo, etc.) cancels auth mode
                    let mut sess = session.lock().await;
                    if let Some(thread) = sess.threads.get_mut(&thread_id) {
                        thread.pending_auth = None;
                    }
                    // Fall through to normal handling
                }
            }
        }

        tracing::debug!(
            "Received message from {} on {} ({} chars)",
            message.user_id,
            message.channel,
            message.content.len()
        );

        // Process based on submission type
        let result = match submission {
            Submission::UserInput { content } => {
                self.process_user_input(message, session, thread_id, &content)
                    .await
            }
            Submission::SystemCommand { command, args } => {
                self.handle_system_command(&command, &args).await
            }
            Submission::Undo => self.process_undo(session, thread_id).await,
            Submission::Redo => self.process_redo(session, thread_id).await,
            Submission::Interrupt => self.process_interrupt(session, thread_id).await,
            Submission::Compact => self.process_compact(session, thread_id).await,
            Submission::Clear => self.process_clear(session, thread_id).await,
            Submission::NewThread => self.process_new_thread(message).await,
            Submission::Heartbeat => self.process_heartbeat().await,
            Submission::Summarize => self.process_summarize(session, thread_id).await,
            Submission::Suggest => self.process_suggest(session, thread_id).await,
            Submission::JobStatus { job_id } => {
                self.process_job_status(&message.user_id, job_id.as_deref())
                    .await
            }
            Submission::JobCancel { job_id } => {
                self.process_job_cancel(&message.user_id, &job_id).await
            }
            Submission::Quit => return Ok(None),
            Submission::SwitchThread { thread_id: target } => {
                self.process_switch_thread(message, target).await
            }
            Submission::Resume { checkpoint_id } => {
                self.process_resume(session, thread_id, checkpoint_id).await
            }
            Submission::ExecApproval {
                request_id,
                approved,
                always,
            } => {
                self.process_approval(
                    message,
                    session,
                    thread_id,
                    Some(request_id),
                    approved,
                    always,
                )
                .await
            }
            Submission::ApprovalResponse { approved, always } => {
                self.process_approval(message, session, thread_id, None, approved, always)
                    .await
            }
        };

        // Convert SubmissionResult to response string
        match result? {
            SubmissionResult::Response { content } => {
                // Suppress silent replies (e.g. from group chat "nothing to say" responses)
                if crate::llm::is_silent_reply(&content) {
                    tracing::debug!("Suppressing silent reply token");
                    Ok(Some(String::new()))
                } else {
                    Ok(Some(content))
                }
            }
            SubmissionResult::Ok { message } => Ok(message),
            SubmissionResult::Error { message } => Ok(Some(format!("Error: {}", message))),
            SubmissionResult::Interrupted => Ok(Some("Interrupted.".into())),
            SubmissionResult::NeedApproval {
                request_id,
                tool_name,
                description,
                parameters,
            } => {
                // Each channel renders the approval prompt via send_status.
                // Web gateway shows an inline card, REPL prints a formatted prompt, etc.
                let _ = self
                    .channels
                    .send_status(
                        &message.channel,
                        StatusUpdate::ApprovalNeeded {
                            request_id: request_id.to_string(),
                            tool_name,
                            description,
                            parameters,
                        },
                        &message.metadata,
                    )
                    .await;

                // Empty string signals the caller to skip respond() (no duplicate text)
                Ok(Some(String::new()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::canonicalize_external_thread_id;
    use super::normalize_approval_submission_for_thread;
    use super::truncate_for_preview;
    use crate::agent::session::ThreadState;
    use crate::agent::submission::Submission;
    use uuid::Uuid;

    #[test]
    fn test_truncate_short_input() {
        assert_eq!(truncate_for_preview("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_empty_input() {
        assert_eq!(truncate_for_preview("", 10), "");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate_for_preview("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_over_limit() {
        let result = truncate_for_preview("hello world, this is long", 10);
        assert!(result.ends_with("..."));
        // "hello worl" = 10 chars + "..."
        assert_eq!(result, "hello worl...");
    }

    #[test]
    fn test_truncate_collapses_newlines() {
        let result = truncate_for_preview("line1\nline2\nline3", 100);
        assert!(!result.contains('\n'));
        assert_eq!(result, "line1 line2 line3");
    }

    #[test]
    fn test_truncate_collapses_whitespace() {
        let result = truncate_for_preview("hello   world", 100);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_truncate_multibyte_utf8() {
        // Each emoji is 4 bytes. Truncating at char boundary must not panic.
        let input = "😀😁😂🤣😃😄😅😆😉😊";
        let result = truncate_for_preview(input, 5);
        assert!(result.ends_with("..."));
        // First 5 chars = 5 emoji
        assert_eq!(result, "😀😁😂🤣😃...");
    }

    #[test]
    fn test_truncate_cjk_characters() {
        // CJK chars are 3 bytes each in UTF-8.
        let input = "你好世界测试数据很长的字符串";
        let result = truncate_for_preview(input, 4);
        assert_eq!(result, "你好世界...");
    }

    #[test]
    fn test_truncate_mixed_multibyte_and_ascii() {
        let input = "hello 世界 foo";
        let result = truncate_for_preview(input, 8);
        // 'h','e','l','l','o',' ','世','界' = 8 chars
        assert_eq!(result, "hello 世界...");
    }

    #[test]
    fn test_canonicalize_external_thread_id_preserves_uuid() {
        let id = Uuid::new_v4().to_string();
        let canonical = canonicalize_external_thread_id("telegram", "user-1", &id);
        assert_eq!(canonical, id);
    }

    #[test]
    fn test_canonicalize_external_thread_id_is_stable() {
        let id1 = canonicalize_external_thread_id("telegram", "user-1", "123456789");
        let id2 = canonicalize_external_thread_id("telegram", "user-1", "123456789");
        assert_eq!(id1, id2);
        assert!(Uuid::parse_str(&id1).is_ok());
    }

    #[test]
    fn test_canonicalize_external_thread_id_scopes_by_channel_and_user() {
        let base = canonicalize_external_thread_id("telegram", "user-1", "42");
        let diff_channel = canonicalize_external_thread_id("signal", "user-1", "42");
        let diff_user = canonicalize_external_thread_id("telegram", "user-2", "42");

        assert_ne!(base, diff_channel);
        assert_ne!(base, diff_user);
    }

    #[test]
    fn test_approval_alias_becomes_user_input_when_not_awaiting() {
        let submission = Submission::ApprovalResponse {
            approved: true,
            always: false,
        };
        let normalized =
            normalize_approval_submission_for_thread(submission, "ok", Some(ThreadState::Idle))
                .expect("should normalize");
        assert!(matches!(
            normalized,
            Submission::UserInput { content } if content == "ok"
        ));
    }

    #[test]
    fn test_approval_alias_kept_when_awaiting() {
        let submission = Submission::ApprovalResponse {
            approved: true,
            always: false,
        };
        let normalized = normalize_approval_submission_for_thread(
            submission.clone(),
            "yes",
            Some(ThreadState::AwaitingApproval),
        )
        .expect("should preserve approval");
        assert!(matches!(
            normalized,
            Submission::ApprovalResponse {
                approved: true,
                always: false
            }
        ));
    }

    #[test]
    fn test_slash_approval_outside_pending_returns_error() {
        let submission = Submission::ApprovalResponse {
            approved: true,
            always: false,
        };
        let err = normalize_approval_submission_for_thread(
            submission,
            "/approve",
            Some(ThreadState::Idle),
        )
        .expect_err("slash approval should error without pending request");
        assert_eq!(err, "Error: No pending approval request.");
    }
}
