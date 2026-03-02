//! Conversation-related ConversationStore implementation for LibSqlBackend.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use libsql::params;
use uuid::Uuid;

use super::{LibSqlBackend, fmt_ts, get_i64, get_json, get_opt_text, get_text, get_ts, opt_text};
use crate::db::ConversationStore;
use crate::error::DatabaseError;
use crate::history::{ConversationMessage, ConversationSummary};

#[async_trait]
impl ConversationStore for LibSqlBackend {
    async fn create_conversation(
        &self,
        channel: &str,
        user_id: &str,
        thread_id: Option<&str>,
    ) -> Result<Uuid, DatabaseError> {
        let conn = self.connect().await?;
        let id = Uuid::new_v4();
        conn.execute(
            "INSERT INTO conversations (id, channel, user_id, thread_id) VALUES (?1, ?2, ?3, ?4)",
            params![id.to_string(), channel, user_id, opt_text(thread_id)],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(id)
    }

    async fn touch_conversation(&self, id: Uuid) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        let now = fmt_ts(&Utc::now());
        conn.execute(
            "UPDATE conversations SET last_activity = ?2 WHERE id = ?1",
            params![id.to_string(), now],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn add_conversation_message(
        &self,
        conversation_id: Uuid,
        role: &str,
        content: &str,
    ) -> Result<Uuid, DatabaseError> {
        let conn = self.connect().await?;
        let id = Uuid::new_v4();
        let now = fmt_ts(&Utc::now());
        conn.execute(
                "INSERT INTO conversation_messages (id, conversation_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![id.to_string(), conversation_id.to_string(), role, content, now],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        self.touch_conversation(conversation_id).await?;
        Ok(id)
    }

    async fn ensure_conversation(
        &self,
        id: Uuid,
        channel: &str,
        user_id: &str,
        thread_id: Option<&str>,
    ) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        let now = fmt_ts(&Utc::now());
        conn.execute(
            r#"
                INSERT INTO conversations (id, channel, user_id, thread_id)
                VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT (id) DO UPDATE SET last_activity = ?5
                "#,
            params![id.to_string(), channel, user_id, opt_text(thread_id), now],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn list_conversations_with_preview(
        &self,
        user_id: &str,
        channel: &str,
        limit: i64,
    ) -> Result<Vec<ConversationSummary>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT
                    c.id,
                    c.started_at,
                    c.last_activity,
                    c.metadata,
                    (SELECT COUNT(*) FROM conversation_messages m WHERE m.conversation_id = c.id AND m.role = 'user') AS message_count,
                    (SELECT substr(m2.content, 1, 100)
                     FROM conversation_messages m2
                     WHERE m2.conversation_id = c.id AND m2.role = 'user'
                     ORDER BY m2.created_at ASC, m2.rowid ASC
                     LIMIT 1
                    ) AS title
                FROM conversations c
                WHERE c.user_id = ?1 AND c.channel = ?2
                ORDER BY c.last_activity DESC
                LIMIT ?3
                "#,
                params![user_id, channel, limit],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut results = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            let metadata = get_json(&row, 3);
            let thread_type = metadata
                .get("thread_type")
                .and_then(|v| v.as_str())
                .map(String::from);
            results.push(ConversationSummary {
                id: row
                    .get::<String>(0)
                    .unwrap_or_default()
                    .parse()
                    .unwrap_or_default(),
                started_at: get_ts(&row, 1),
                last_activity: get_ts(&row, 2),
                message_count: get_i64(&row, 4),
                title: get_opt_text(&row, 5),
                thread_type,
            });
        }
        Ok(results)
    }

    async fn get_or_create_assistant_conversation(
        &self,
        user_id: &str,
        channel: &str,
    ) -> Result<Uuid, DatabaseError> {
        let conn = self.connect().await?;
        // Try to find existing
        let mut rows = conn
            .query(
                r#"
                SELECT id FROM conversations
                WHERE user_id = ?1 AND channel = ?2
                  AND json_extract(metadata, '$.thread_type') = 'assistant'
                LIMIT 1
                "#,
                params![user_id, channel],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        if let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            let id_str: String = row.get(0).unwrap_or_default();
            return id_str
                .parse()
                .map_err(|_| DatabaseError::Serialization("Invalid UUID".to_string()));
        }

        // Create new
        let id = Uuid::new_v4();
        let metadata = serde_json::json!({"thread_type": "assistant", "title": "Assistant"});
        conn.execute(
            "INSERT INTO conversations (id, channel, user_id, metadata) VALUES (?1, ?2, ?3, ?4)",
            params![id.to_string(), channel, user_id, metadata.to_string()],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(id)
    }

    async fn create_conversation_with_metadata(
        &self,
        channel: &str,
        user_id: &str,
        metadata: &serde_json::Value,
    ) -> Result<Uuid, DatabaseError> {
        let conn = self.connect().await?;
        let id = Uuid::new_v4();
        conn.execute(
            "INSERT INTO conversations (id, channel, user_id, metadata) VALUES (?1, ?2, ?3, ?4)",
            params![id.to_string(), channel, user_id, metadata.to_string()],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(id)
    }

    async fn list_conversation_messages_paginated(
        &self,
        conversation_id: Uuid,
        before: Option<DateTime<Utc>>,
        limit: i64,
    ) -> Result<(Vec<ConversationMessage>, bool), DatabaseError> {
        let conn = self.connect().await?;
        let fetch_limit = limit + 1;
        let cid = conversation_id.to_string();

        let mut rows = if let Some(before_ts) = before {
            conn.query(
                r#"
                    SELECT id, role, content, created_at
                    FROM conversation_messages
                    WHERE conversation_id = ?1 AND created_at < ?2
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?3
                    "#,
                params![cid, fmt_ts(&before_ts), fetch_limit],
            )
            .await
        } else {
            conn.query(
                r#"
                    SELECT id, role, content, created_at
                    FROM conversation_messages
                    WHERE conversation_id = ?1
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?2
                    "#,
                params![cid, fetch_limit],
            )
            .await
        }
        .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut all = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            all.push(ConversationMessage {
                id: get_text(&row, 0).parse().unwrap_or_default(),
                role: get_text(&row, 1),
                content: get_text(&row, 2),
                created_at: get_ts(&row, 3),
            });
        }

        let has_more = all.len() as i64 > limit;
        all.truncate(limit as usize);
        all.reverse(); // oldest first
        Ok((all, has_more))
    }

    async fn update_conversation_metadata_field(
        &self,
        id: Uuid,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        // SQLite: use json_patch to merge the key
        let patch = serde_json::json!({ key: value });
        conn.execute(
            "UPDATE conversations SET metadata = json_patch(metadata, ?2) WHERE id = ?1",
            params![id.to_string(), patch.to_string()],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn get_conversation_metadata(
        &self,
        id: Uuid,
    ) -> Result<Option<serde_json::Value>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT metadata FROM conversations WHERE id = ?1",
                params![id.to_string()],
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

    async fn list_conversation_messages(
        &self,
        conversation_id: Uuid,
    ) -> Result<Vec<ConversationMessage>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id, role, content, created_at
                FROM conversation_messages
                WHERE conversation_id = ?1
                ORDER BY created_at ASC, rowid ASC
                "#,
                params![conversation_id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut messages = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            messages.push(ConversationMessage {
                id: get_text(&row, 0).parse().unwrap_or_default(),
                role: get_text(&row, 1),
                content: get_text(&row, 2),
                created_at: get_ts(&row, 3),
            });
        }
        Ok(messages)
    }

    async fn conversation_belongs_to_user(
        &self,
        conversation_id: Uuid,
        user_id: &str,
    ) -> Result<bool, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT 1 FROM conversations WHERE id = ?1 AND user_id = ?2",
                libsql::params![conversation_id.to_string(), user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        let found = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(found.is_some())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use serde_json::json;

    use crate::db::libsql::LibSqlBackend;
    use crate::db::{ConversationStore, Database};

    async fn test_backend() -> (LibSqlBackend, tempfile::TempDir) {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("conversations-tests.db");
        let backend = LibSqlBackend::new_local(&db_path).await.unwrap();
        backend.run_migrations().await.unwrap();
        (backend, tempdir)
    }

    #[tokio::test]
    async fn update_conversation_metadata_field_preserves_other_top_level_keys() {
        let (backend, _tempdir) = test_backend().await;
        let conversation_id = backend
            .create_conversation_with_metadata(
                "web",
                "user-1",
                &json!({
                    "thread_type": "assistant",
                    "title": "Assistant",
                    "prefs": { "theme": "light", "density": "comfortable" }
                }),
            )
            .await
            .unwrap();

        backend
            .update_conversation_metadata_field(
                conversation_id,
                "prefs",
                &json!({ "theme": "dark" }),
            )
            .await
            .unwrap();

        let metadata = backend
            .get_conversation_metadata(conversation_id)
            .await
            .unwrap()
            .unwrap();

        assert_eq!(metadata.get("thread_type"), Some(&json!("assistant")));
        assert_eq!(metadata.get("title"), Some(&json!("Assistant")));
        assert_eq!(
            metadata.get("prefs"),
            Some(&json!({ "theme": "dark", "density": "comfortable" }))
        );
    }

    #[tokio::test]
    async fn get_or_create_assistant_conversation_is_idempotent_per_user_and_channel() {
        let (backend, _tempdir) = test_backend().await;

        let first = backend
            .get_or_create_assistant_conversation("user-1", "web")
            .await
            .unwrap();
        let second = backend
            .get_or_create_assistant_conversation("user-1", "web")
            .await
            .unwrap();
        let other_channel = backend
            .get_or_create_assistant_conversation("user-1", "signal")
            .await
            .unwrap();

        assert_eq!(first, second);
        assert_ne!(first, other_channel);

        let metadata = backend
            .get_conversation_metadata(first)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(metadata.get("thread_type"), Some(&json!("assistant")));
    }

    #[tokio::test]
    async fn list_conversation_messages_paginated_returns_stable_pages() {
        let (backend, _tempdir) = test_backend().await;
        let conversation_id = backend
            .create_conversation("web", "user-1", None)
            .await
            .unwrap();

        backend
            .add_conversation_message(conversation_id, "user", "m1")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(3)).await;
        backend
            .add_conversation_message(conversation_id, "assistant", "m2")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(3)).await;
        backend
            .add_conversation_message(conversation_id, "user", "m3")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(3)).await;
        backend
            .add_conversation_message(conversation_id, "assistant", "m4")
            .await
            .unwrap();

        let (page_one, has_more_one) = backend
            .list_conversation_messages_paginated(conversation_id, None, 2)
            .await
            .unwrap();
        assert_eq!(
            page_one
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["m3", "m4"]
        );
        assert!(has_more_one);

        let before = page_one.first().map(|m| m.created_at);
        let (page_two, has_more_two) = backend
            .list_conversation_messages_paginated(conversation_id, before, 2)
            .await
            .unwrap();
        assert_eq!(
            page_two
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>(),
            vec!["m1", "m2"]
        );
        assert!(!has_more_two);
    }

    #[tokio::test]
    async fn conversation_belongs_to_user_checks_owner() {
        let (backend, _tempdir) = test_backend().await;

        let a = backend
            .create_conversation("web", "owner-a", None)
            .await
            .unwrap();
        let b = backend
            .create_conversation("web", "owner-b", None)
            .await
            .unwrap();

        assert!(
            backend
                .conversation_belongs_to_user(a, "owner-a")
                .await
                .unwrap()
        );
        assert!(
            !backend
                .conversation_belongs_to_user(a, "owner-b")
                .await
                .unwrap()
        );
        assert!(
            backend
                .conversation_belongs_to_user(b, "owner-b")
                .await
                .unwrap()
        );
    }
}
