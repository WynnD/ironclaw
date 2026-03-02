//! WASM tool runtime for managing compiled components.
//!
//! Follows the principle: compile once at registration, instantiate fresh per execution.
//! This matches NEAR blockchain patterns for deterministic, isolated execution.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use wasmtime::Store;
use wasmtime::component::Linker;
use wasmtime::{Config, Engine, OptLevel};
use wasmtime_wasi::{ResourceTable, WasiCtx, WasiCtxBuilder, WasiView};

use crate::tools::wasm::error::WasmError;
use crate::tools::wasm::limits::{FuelConfig, ResourceLimits};

// Generate component model bindings from the shared tool WIT contract so we can
// call `description()` and `schema()` during module preparation.
wasmtime::component::bindgen!({
    path: "wit/tool.wit",
    world: "sandboxed-tool",
    async: false,
    with: {},
});

/// Default epoch tick interval. Each tick increments the engine's epoch counter,
/// which causes any store with an expired epoch deadline to trap.
pub const EPOCH_TICK_INTERVAL: Duration = Duration::from_millis(500);

/// Configuration for the WASM runtime.
#[derive(Debug, Clone)]
pub struct WasmRuntimeConfig {
    /// Default resource limits for tools.
    pub default_limits: ResourceLimits,
    /// Fuel configuration.
    pub fuel_config: FuelConfig,
    /// Whether to cache compiled modules.
    pub cache_compiled: bool,
    /// Directory for compiled module cache.
    pub cache_dir: Option<PathBuf>,
    /// Cranelift optimization level.
    pub optimization_level: OptLevel,
}

impl Default for WasmRuntimeConfig {
    fn default() -> Self {
        Self {
            default_limits: ResourceLimits::default(),
            fuel_config: FuelConfig::default(),
            cache_compiled: true,
            cache_dir: None,
            optimization_level: OptLevel::Speed,
        }
    }
}

impl WasmRuntimeConfig {
    /// Create a minimal config for testing.
    pub fn for_testing() -> Self {
        Self {
            default_limits: ResourceLimits::default()
                .with_memory(1024 * 1024) // 1 MB
                .with_fuel(100_000)
                .with_timeout(Duration::from_secs(5)),
            fuel_config: FuelConfig::with_limit(100_000),
            cache_compiled: false,
            cache_dir: None,
            optimization_level: OptLevel::None, // Faster compilation for tests
        }
    }
}

/// A compiled WASM component ready for instantiation.
///
/// Contains the pre-compiled component plus cached metadata extracted
/// from the component during preparation. Stores the compiled `Component`
/// directly so instantiation doesn't require recompilation.
pub struct PreparedModule {
    /// Tool name.
    pub name: String,
    /// Tool description (cached from component).
    pub description: String,
    /// Parameter schema JSON (cached from component).
    pub schema: serde_json::Value,
    /// Pre-compiled component (cheaply cloneable via internal Arc).
    component: wasmtime::component::Component,
    /// Resource limits for this tool.
    pub limits: ResourceLimits,
}

impl PreparedModule {
    /// Get the pre-compiled component for instantiation.
    pub fn component(&self) -> &wasmtime::component::Component {
        &self.component
    }
}

impl std::fmt::Debug for PreparedModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedModule")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("limits", &self.limits)
            .finish()
    }
}

/// WASM tool runtime.
///
/// Manages the Wasmtime engine and a cache of prepared modules.
pub struct WasmToolRuntime {
    /// Wasmtime engine with configured settings.
    engine: Engine,
    /// Runtime configuration.
    config: WasmRuntimeConfig,
    /// Cache of prepared modules by name.
    modules: RwLock<HashMap<String, Arc<PreparedModule>>>,
}

impl WasmToolRuntime {
    /// Create a new runtime with the given configuration.
    pub fn new(config: WasmRuntimeConfig) -> Result<Self, WasmError> {
        let mut wasmtime_config = Config::new();

        // Enable fuel consumption for CPU limiting
        if config.fuel_config.enabled {
            wasmtime_config.consume_fuel(true);
        }

        // Enable epoch interruption as a backup timeout mechanism
        wasmtime_config.epoch_interruption(true);

        // Enable component model (WASI Preview 2)
        wasmtime_config.wasm_component_model(true);

        // Disable threads (simplifies security model)
        wasmtime_config.wasm_threads(false);

        // Set optimization level
        wasmtime_config.cranelift_opt_level(config.optimization_level);

        // Disable debug info in production for smaller modules
        wasmtime_config.debug_info(false);

        // Enable persistent compilation cache. Wasmtime serializes compiled native
        // code to disk (~/.cache/wasmtime by default), so subsequent startups
        // deserialize instead of recompiling — typically 10-50x faster.
        if let Err(e) = wasmtime_config.cache_config_load_default() {
            tracing::warn!("Failed to enable wasmtime compilation cache: {}", e);
        }

        let engine = Engine::new(&wasmtime_config).map_err(|e| {
            WasmError::EngineCreationFailed(format!("Failed to create Wasmtime engine: {}", e))
        })?;

        // Spawn a background thread that periodically increments the engine's
        // epoch counter. Without this, epoch_deadline_trap() never fires and
        // WASM modules can spin indefinitely even with a deadline set.
        let ticker_engine = engine.clone();
        std::thread::Builder::new()
            .name("wasm-epoch-ticker".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(EPOCH_TICK_INTERVAL);
                    ticker_engine.increment_epoch();
                }
            })
            .map_err(|e| {
                WasmError::EngineCreationFailed(format!(
                    "Failed to spawn epoch ticker thread: {}",
                    e
                ))
            })?;

        Ok(Self {
            engine,
            config,
            modules: RwLock::new(HashMap::new()),
        })
    }

    /// Get the Wasmtime engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Get the runtime configuration.
    pub fn config(&self) -> &WasmRuntimeConfig {
        &self.config
    }

    /// Prepare a WASM component for execution.
    ///
    /// This validates and compiles the component, extracting metadata.
    /// The compiled component is cached for fast instantiation.
    pub async fn prepare(
        &self,
        name: &str,
        wasm_bytes: &[u8],
        limits: Option<ResourceLimits>,
    ) -> Result<Arc<PreparedModule>, WasmError> {
        // Check if already prepared
        if let Some(module) = self.modules.read().await.get(name) {
            return Ok(Arc::clone(module));
        }

        let name = name.to_string();
        let wasm_bytes = wasm_bytes.to_vec();
        let engine = self.engine.clone();
        let default_limits = self.config.default_limits.clone();

        // Compile in blocking task (Wasmtime compilation is synchronous)
        let prepared = tokio::task::spawn_blocking(move || {
            // Validate and compile the component
            let component = wasmtime::component::Component::new(&engine, &wasm_bytes)
                .map_err(|e| WasmError::CompilationFailed(e.to_string()))?;

            // Instantiate once to read metadata from exports (`description`, `schema`).
            let (description, schema) = match extract_tool_metadata(&engine, &component) {
                Ok(metadata) => metadata,
                Err(e) => {
                    tracing::warn!(
                        name = %name,
                        error = %e,
                        "Failed to extract WASM tool metadata; using fallback description/schema"
                    );
                    (default_tool_description(), default_tool_schema())
                }
            };

            Ok::<_, WasmError>(PreparedModule {
                name: name.clone(),
                description,
                schema,
                component,
                limits: limits.unwrap_or(default_limits),
            })
        })
        .await
        .map_err(|e| WasmError::ExecutionPanicked(format!("Preparation task panicked: {}", e)))??;

        let prepared = Arc::new(prepared);

        // Cache the prepared module
        if self.config.cache_compiled {
            self.modules
                .write()
                .await
                .insert(prepared.name.clone(), Arc::clone(&prepared));
        }

        tracing::info!(
            name = %prepared.name,
            "Prepared WASM tool for execution"
        );

        Ok(prepared)
    }

    /// Get a prepared module by name.
    pub async fn get(&self, name: &str) -> Option<Arc<PreparedModule>> {
        self.modules.read().await.get(name).cloned()
    }

    /// Remove a prepared module from the cache.
    pub async fn remove(&self, name: &str) -> Option<Arc<PreparedModule>> {
        self.modules.write().await.remove(name)
    }

    /// List all prepared module names.
    pub async fn list(&self) -> Vec<String> {
        self.modules.read().await.keys().cloned().collect()
    }

    /// Clear all cached modules.
    pub async fn clear(&self) {
        self.modules.write().await.clear();
    }
}

/// Minimal store data for metadata extraction.
///
/// Metadata methods should be pure, but the world imports host+WASI interfaces,
/// so instantiation still requires these to be wired.
struct MetadataStoreData {
    wasi: WasiCtx,
    table: ResourceTable,
}

impl MetadataStoreData {
    fn new() -> Self {
        Self {
            wasi: WasiCtxBuilder::new().build(),
            table: ResourceTable::new(),
        }
    }
}

impl WasiView for MetadataStoreData {
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.wasi
    }

    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
}

impl near::agent::host::Host for MetadataStoreData {
    fn log(&mut self, _level: near::agent::host::LogLevel, _message: String) {}

    fn now_millis(&mut self) -> u64 {
        0
    }

    fn workspace_read(&mut self, _path: String) -> Option<String> {
        None
    }

    fn http_request(
        &mut self,
        _method: String,
        _url: String,
        _headers_json: String,
        _body: Option<Vec<u8>>,
        _timeout_ms: Option<u32>,
    ) -> Result<near::agent::host::HttpResponse, String> {
        Err("http_request is unavailable during metadata extraction".to_string())
    }

    fn tool_invoke(&mut self, _alias: String, _params_json: String) -> Result<String, String> {
        Err("tool_invoke is unavailable during metadata extraction".to_string())
    }

    fn secret_exists(&mut self, _name: String) -> bool {
        false
    }
}

fn default_tool_description() -> String {
    "WASM sandboxed tool".to_string()
}

fn default_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    })
}

fn normalize_tool_description(description: String) -> String {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        default_tool_description()
    } else {
        trimmed.to_string()
    }
}

/// Parse and validate the schema string exported by a WASM tool.
pub(crate) fn parse_tool_schema(schema_json: &str) -> Result<serde_json::Value, WasmError> {
    let schema: serde_json::Value = serde_json::from_str(schema_json).map_err(|e| {
        WasmError::InvalidResponseJson(format!("Tool schema export is not valid JSON: {}", e))
    })?;

    if !schema.is_object() {
        return Err(WasmError::InvalidResponseJson(
            "Tool schema export must be a JSON object".to_string(),
        ));
    }

    Ok(schema)
}

/// Extract description + parameter schema from a compiled component by calling
/// the exported metadata methods in the WIT `tool` interface.
fn extract_tool_metadata(
    engine: &Engine,
    component: &wasmtime::component::Component,
) -> Result<(String, serde_json::Value), WasmError> {
    let mut linker = Linker::new(engine);
    wasmtime_wasi::add_to_linker_sync(&mut linker)
        .map_err(|e| WasmError::ConfigError(format!("Failed to add WASI linker: {}", e)))?;
    near::agent::host::add_to_linker(&mut linker, |state| state).map_err(|e| {
        WasmError::ConfigError(format!(
            "Failed to add host linker for metadata extraction: {}",
            e
        ))
    })?;

    let mut store = Store::new(engine, MetadataStoreData::new());
    let instance = SandboxedTool::instantiate(&mut store, component, &linker).map_err(|e| {
        WasmError::InstantiationFailed(format!(
            "Failed to instantiate WASM component for metadata extraction: {}",
            e
        ))
    })?;

    let tool_iface = instance.near_agent_tool();
    let description = tool_iface.call_description(&mut store).map_err(|e| {
        WasmError::InvalidResponseJson(format!("WASM description() export call failed: {}", e))
    })?;
    let schema_json = tool_iface.call_schema(&mut store).map_err(|e| {
        WasmError::InvalidResponseJson(format!("WASM schema() export call failed: {}", e))
    })?;
    let schema = parse_tool_schema(&schema_json)?;

    Ok((normalize_tool_description(description), schema))
}

impl std::fmt::Debug for WasmToolRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmToolRuntime")
            .field("config", &self.config)
            .field("modules", &"<RwLock<HashMap>>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::tools::wasm::limits::ResourceLimits;
    use crate::tools::wasm::runtime::{
        WasmRuntimeConfig, WasmToolRuntime, normalize_tool_description, parse_tool_schema,
    };

    #[test]
    fn test_runtime_config_default() {
        let config = WasmRuntimeConfig::default();
        assert!(config.cache_compiled);
        assert!(config.fuel_config.enabled);
    }

    #[test]
    fn test_runtime_config_for_testing() {
        let config = WasmRuntimeConfig::for_testing();
        assert!(!config.cache_compiled);
        assert_eq!(config.default_limits.memory_bytes, 1024 * 1024);
    }

    #[test]
    fn test_runtime_creation() {
        let config = WasmRuntimeConfig::for_testing();
        let runtime = WasmToolRuntime::new(config).unwrap();
        // Engine was created successfully, which validates the config
        assert!(runtime.config().fuel_config.enabled);
    }

    #[tokio::test]
    async fn test_module_cache_operations() {
        let config = WasmRuntimeConfig::for_testing();
        let runtime = WasmToolRuntime::new(config).unwrap();

        // Initially empty
        assert!(runtime.list().await.is_empty());
        assert!(runtime.get("test").await.is_none());
    }

    #[test]
    fn test_prepared_module_limits() {
        let limits = ResourceLimits::default()
            .with_memory(5 * 1024 * 1024)
            .with_fuel(500_000);

        assert_eq!(limits.memory_bytes, 5 * 1024 * 1024);
        assert_eq!(limits.fuel, 500_000);
    }

    /// The WASM runtime (Wasmtime engine) must initialise successfully even
    /// when no tools directory exists on disk. The engine only configures the
    /// compiler and epoch ticker — loading modules from a directory is a
    /// separate step. Regression test for a bug where the runtime was gated
    /// on `tools_dir.exists()`, causing extensions installed after startup
    /// (e.g. via the web UI) to fail with "WASM runtime not available".
    #[test]
    fn test_runtime_creation_without_tools_dir() {
        let config = WasmRuntimeConfig::for_testing();
        // Runtime should succeed even though no tools directory exists.
        let runtime = WasmToolRuntime::new(config).expect("runtime should init without tools dir");
        assert!(runtime.config().fuel_config.enabled);
    }

    #[test]
    fn test_parse_tool_schema_valid_object() {
        let schema =
            parse_tool_schema(r#"{"type":"object","properties":{"action":{"type":"string"}}}"#)
                .expect("schema parse should succeed");
        assert_eq!(schema["type"], json!("object"));
        assert!(schema["properties"]["action"].is_object());
    }

    #[test]
    fn test_parse_tool_schema_invalid_json() {
        let err = parse_tool_schema("{not-json").expect_err("invalid json should be rejected");
        assert!(err.to_string().contains("not valid JSON"));
    }

    #[test]
    fn test_parse_tool_schema_rejects_non_object() {
        let err =
            parse_tool_schema(r#"["array"]"#).expect_err("non-object schemas should be rejected");
        assert!(err.to_string().contains("must be a JSON object"));
    }

    #[test]
    fn test_normalize_tool_description_fallback() {
        assert_eq!(
            normalize_tool_description("  ".to_string()),
            "WASM sandboxed tool"
        );
        assert_eq!(
            normalize_tool_description("  Google Calendar tool  ".to_string()),
            "Google Calendar tool"
        );
    }
}
