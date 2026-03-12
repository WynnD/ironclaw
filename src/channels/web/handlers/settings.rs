//! Settings API handlers.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};

use crate::channels::web::server::GatewayState;
use crate::channels::web::types::*;

pub async fn settings_list_handler(
    State(state): State<Arc<GatewayState>>,
) -> Result<Json<SettingsListResponse>, StatusCode> {
    let store = state
        .store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let rows = store.list_settings(&state.user_id).await.map_err(|e| {
        tracing::error!("Failed to list settings: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let settings = rows
        .into_iter()
        .map(|r| SettingResponse {
            key: r.key,
            value: r.value,
            updated_at: r.updated_at.to_rfc3339(),
        })
        .collect();

    Ok(Json(SettingsListResponse { settings }))
}

pub async fn settings_get_handler(
    State(state): State<Arc<GatewayState>>,
    Path(key): Path<String>,
) -> Result<Json<SettingResponse>, StatusCode> {
    let store = state
        .store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let row = store
        .get_setting_full(&state.user_id, &key)
        .await
        .map_err(|e| {
            tracing::error!("Failed to get setting '{}': {}", key, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(SettingResponse {
        key: row.key,
        value: row.value,
        updated_at: row.updated_at.to_rfc3339(),
    }))
}

pub async fn settings_set_handler(
    State(state): State<Arc<GatewayState>>,
    Path(key): Path<String>,
    Json(body): Json<SettingWriteRequest>,
) -> Result<StatusCode, StatusCode> {
    let store = state
        .store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    store
        .set_setting(&state.user_id, &key, &body.value)
        .await
        .map_err(|e| {
            tracing::error!("Failed to set setting '{}': {}", key, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(StatusCode::NO_CONTENT)
}

pub async fn settings_delete_handler(
    State(state): State<Arc<GatewayState>>,
    Path(key): Path<String>,
) -> Result<StatusCode, StatusCode> {
    let store = state
        .store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    store
        .delete_setting(&state.user_id, &key)
        .await
        .map_err(|e| {
            tracing::error!("Failed to delete setting '{}': {}", key, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(StatusCode::NO_CONTENT)
}

pub async fn settings_export_handler(
    State(state): State<Arc<GatewayState>>,
) -> Result<Json<SettingsExportResponse>, StatusCode> {
    let store = state
        .store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let settings = store.get_all_settings(&state.user_id).await.map_err(|e| {
        tracing::error!("Failed to export settings: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(SettingsExportResponse { settings }))
}

pub async fn settings_import_handler(
    State(state): State<Arc<GatewayState>>,
    Json(body): Json<SettingsImportRequest>,
) -> Result<StatusCode, StatusCode> {
    let store = state
        .store
        .as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    store
        .set_all_settings(&state.user_id, &body.settings)
        .await
        .map_err(|e| {
            tracing::error!("Failed to import settings: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use axum::{
        Json,
        extract::{Path, State},
        http::StatusCode,
    };
    use serde_json::json;

    use super::*;
    use crate::channels::web::server::{GatewayState, RateLimiter};
    use crate::channels::web::sse::SseManager;
    use crate::db::Database;
    use crate::db::libsql::LibSqlBackend;

    async fn test_backend() -> (Arc<dyn Database>, tempfile::TempDir) {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("settings-handler-tests.db");
        let backend = LibSqlBackend::new_local(&db_path).await.unwrap();
        backend.run_migrations().await.unwrap();
        (Arc::new(backend), tempdir)
    }

    fn make_state(store: Option<Arc<dyn Database>>, user_id: &str) -> Arc<GatewayState> {
        Arc::new(GatewayState {
            msg_tx: tokio::sync::RwLock::new(None),
            sse: SseManager::new(),
            workspace: None,
            session_manager: None,
            log_broadcaster: None,
            log_level_handle: None,
            extension_manager: None,
            tool_registry: None,
            store,
            secrets_store: None,
            job_manager: None,
            prompt_queue: None,
            user_id: user_id.to_string(),
            shutdown_tx: tokio::sync::RwLock::new(None),
            ws_tracker: None,
            llm_provider: None,
            skill_registry: None,
            skill_catalog: None,
            chat_rate_limiter: RateLimiter::new(30, 60),
            registry_entries: Vec::new(),
            cost_guard: None,
            context_monitor: tokio::sync::RwLock::new(None),
            startup_time: std::time::Instant::now(),
            scheduler: None,
        })
    }

    #[tokio::test]
    async fn settings_handlers_scope_to_gateway_user_id() {
        let (store, _tempdir) = test_backend().await;
        store
            .set_setting(
                "gateway-user-123",
                "agent.auto_approved_tools",
                &json!(["shell"]),
            )
            .await
            .unwrap();
        store
            .set_setting("default", "agent.auto_approved_tools", &json!(["http"]))
            .await
            .unwrap();

        let state = make_state(Some(store), "gateway-user-123");

        let export = settings_export_handler(State(state.clone()))
            .await
            .unwrap()
            .0;
        assert_eq!(
            export.settings.get("agent.auto_approved_tools"),
            Some(&json!(["shell"]))
        );

        let missing_default =
            settings_get_handler(State(state.clone()), Path("only.for.default".to_string())).await;
        assert_eq!(missing_default.unwrap_err(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn settings_set_get_delete_roundtrip() {
        let (store, _tempdir) = test_backend().await;
        let state = make_state(Some(store), "gateway-user-123");

        let set_status = settings_set_handler(
            State(state.clone()),
            Path("feature.flag".to_string()),
            Json(SettingWriteRequest { value: json!(true) }),
        )
        .await
        .unwrap();
        assert_eq!(set_status, StatusCode::NO_CONTENT);

        let got = settings_get_handler(State(state.clone()), Path("feature.flag".to_string()))
            .await
            .unwrap()
            .0;
        assert_eq!(got.key, "feature.flag");
        assert_eq!(got.value, json!(true));

        let mut import = HashMap::new();
        import.insert("other.setting".to_string(), json!("x"));
        let import_status = settings_import_handler(
            State(state.clone()),
            Json(SettingsImportRequest { settings: import }),
        )
        .await
        .unwrap();
        assert_eq!(import_status, StatusCode::NO_CONTENT);

        let listed = settings_list_handler(State(state.clone())).await.unwrap().0;
        assert!(listed.settings.iter().any(|s| s.key == "feature.flag"));
        assert!(listed.settings.iter().any(|s| s.key == "other.setting"));

        let delete_status =
            settings_delete_handler(State(state.clone()), Path("feature.flag".to_string()))
                .await
                .unwrap();
        assert_eq!(delete_status, StatusCode::NO_CONTENT);

        let missing = settings_get_handler(State(state), Path("feature.flag".to_string())).await;
        assert_eq!(missing.unwrap_err(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn settings_handlers_return_service_unavailable_without_store() {
        let state = make_state(None, "gateway-user-123");

        assert_eq!(
            settings_list_handler(State(state.clone()))
                .await
                .unwrap_err(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            settings_export_handler(State(state.clone()))
                .await
                .unwrap_err(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            settings_get_handler(State(state), Path("k".to_string()))
                .await
                .unwrap_err(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
