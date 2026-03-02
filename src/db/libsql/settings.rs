//! Settings-related SettingsStore implementation for LibSqlBackend.

use std::collections::HashMap;

use async_trait::async_trait;
use libsql::params;

use super::{LibSqlBackend, fmt_ts, get_i64, get_json, get_text, get_ts};
use crate::db::SettingsStore;
use crate::error::DatabaseError;
use crate::history::SettingRow;

use chrono::Utc;

#[async_trait]
impl SettingsStore for LibSqlBackend {
    async fn get_setting(
        &self,
        user_id: &str,
        key: &str,
    ) -> Result<Option<serde_json::Value>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT value FROM settings WHERE user_id = ?1 AND key = ?2",
                params![user_id, key],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(Some(get_json(&row, 0))),
            None => Ok(None),
        }
    }

    async fn get_setting_full(
        &self,
        user_id: &str,
        key: &str,
    ) -> Result<Option<SettingRow>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT key, value, updated_at FROM settings WHERE user_id = ?1 AND key = ?2",
                params![user_id, key],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(Some(SettingRow {
                key: get_text(&row, 0),
                value: get_json(&row, 1),
                updated_at: get_ts(&row, 2),
            })),
            None => Ok(None),
        }
    }

    async fn set_setting(
        &self,
        user_id: &str,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        let now = fmt_ts(&Utc::now());
        conn.execute(
            r#"
                INSERT INTO settings (user_id, key, value, updated_at)
                VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT (user_id, key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = ?4
                "#,
            params![user_id, key, value.to_string(), now],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn delete_setting(&self, user_id: &str, key: &str) -> Result<bool, DatabaseError> {
        let conn = self.connect().await?;
        let count = conn
            .execute(
                "DELETE FROM settings WHERE user_id = ?1 AND key = ?2",
                params![user_id, key],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(count > 0)
    }

    async fn list_settings(&self, user_id: &str) -> Result<Vec<SettingRow>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT key, value, updated_at FROM settings WHERE user_id = ?1 ORDER BY key",
                params![user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut settings = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            settings.push(SettingRow {
                key: get_text(&row, 0),
                value: get_json(&row, 1),
                updated_at: get_ts(&row, 2),
            });
        }
        Ok(settings)
    }

    async fn get_all_settings(
        &self,
        user_id: &str,
    ) -> Result<HashMap<String, serde_json::Value>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT key, value FROM settings WHERE user_id = ?1",
                params![user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut map = HashMap::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            map.insert(get_text(&row, 0), get_json(&row, 1));
        }
        Ok(map)
    }

    async fn set_all_settings(
        &self,
        user_id: &str,
        settings: &HashMap<String, serde_json::Value>,
    ) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        let now = fmt_ts(&Utc::now());
        conn.execute("BEGIN", ())
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        for (key, value) in settings {
            if let Err(e) = conn
                .execute(
                    r#"
                    INSERT INTO settings (user_id, key, value, updated_at)
                    VALUES (?1, ?2, ?3, ?4)
                    ON CONFLICT (user_id, key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = ?4
                    "#,
                    params![user_id, key.as_str(), value.to_string(), now.as_str()],
                )
                .await
            {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(DatabaseError::Query(e.to_string()));
            }
        }

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn has_settings(&self, user_id: &str) -> Result<bool, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) as cnt FROM settings WHERE user_id = ?1",
                params![user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(get_i64(&row, 0) > 0),
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::json;

    use crate::db::libsql::LibSqlBackend;
    use crate::db::{Database, SettingsStore};

    async fn test_backend() -> (LibSqlBackend, tempfile::TempDir) {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("settings-tests.db");
        let backend = LibSqlBackend::new_local(&db_path).await.unwrap();
        backend.run_migrations().await.unwrap();
        (backend, tempdir)
    }

    #[tokio::test]
    async fn settings_are_isolated_by_user_id() {
        let (backend, _tempdir) = test_backend().await;

        backend
            .set_setting(
                "gateway-user-1",
                "agent.auto_approved_tools",
                &json!(["shell"]),
            )
            .await
            .unwrap();
        backend
            .set_setting("default", "agent.auto_approved_tools", &json!(["none"]))
            .await
            .unwrap();

        let user_one = backend
            .get_setting("gateway-user-1", "agent.auto_approved_tools")
            .await
            .unwrap()
            .unwrap();
        let default_user = backend
            .get_setting("default", "agent.auto_approved_tools")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(user_one, json!(["shell"]));
        assert_eq!(default_user, json!(["none"]));

        let list = backend.list_settings("gateway-user-1").await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].key, "agent.auto_approved_tools");
    }

    #[tokio::test]
    async fn set_all_settings_upserts_without_deleting_other_keys() {
        let (backend, _tempdir) = test_backend().await;

        backend.set_setting("user-1", "a", &json!(1)).await.unwrap();
        backend.set_setting("user-1", "b", &json!(2)).await.unwrap();

        let mut incoming = HashMap::new();
        incoming.insert("a".to_string(), json!(10));
        incoming.insert("c".to_string(), json!(3));
        backend.set_all_settings("user-1", &incoming).await.unwrap();

        let all = backend.get_all_settings("user-1").await.unwrap();
        assert_eq!(all.get("a"), Some(&json!(10)));
        assert_eq!(all.get("b"), Some(&json!(2)));
        assert_eq!(all.get("c"), Some(&json!(3)));
    }

    #[tokio::test]
    async fn has_settings_and_delete_setting_follow_expected_lifecycle() {
        let (backend, _tempdir) = test_backend().await;

        assert!(!backend.has_settings("user-1").await.unwrap());

        backend
            .set_setting("user-1", "feature.flag", &json!(true))
            .await
            .unwrap();
        assert!(backend.has_settings("user-1").await.unwrap());

        let deleted = backend
            .delete_setting("user-1", "feature.flag")
            .await
            .unwrap();
        assert!(deleted);
        assert!(!backend.has_settings("user-1").await.unwrap());

        let deleted_again = backend
            .delete_setting("user-1", "feature.flag")
            .await
            .unwrap();
        assert!(!deleted_again);
    }
}
