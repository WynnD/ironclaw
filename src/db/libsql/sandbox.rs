//! Sandbox-related SandboxStore implementation for LibSqlBackend.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use libsql::params;
use uuid::Uuid;

use super::{
    LibSqlBackend, fmt_opt_ts, fmt_ts, get_i64, get_json, get_opt_bool, get_opt_text, get_opt_ts,
    get_text, get_ts, opt_text,
};
use crate::db::SandboxStore;
use crate::error::DatabaseError;
use crate::history::{JobEventRecord, SandboxJobRecord, SandboxJobSummary};

#[async_trait]
impl SandboxStore for LibSqlBackend {
    async fn save_sandbox_job(&self, job: &SandboxJobRecord) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
                INSERT INTO agent_jobs (
                    id, title, description, status, source, user_id, project_dir,
                    success, failure_reason, created_at, started_at, completed_at
                ) VALUES (?1, ?2, ?3, ?4, 'sandbox', ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                ON CONFLICT (id) DO UPDATE SET
                    status = excluded.status,
                    success = excluded.success,
                    failure_reason = excluded.failure_reason,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at
                "#,
            params![
                job.id.to_string(),
                job.task.as_str(),
                job.credential_grants_json.as_str(),
                job.status.as_str(),
                job.user_id.as_str(),
                job.project_dir.as_str(),
                job.success.map(|b| b as i64),
                opt_text(job.failure_reason.as_deref()),
                fmt_ts(&job.created_at),
                fmt_opt_ts(&job.started_at),
                fmt_opt_ts(&job.completed_at),
            ],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn get_sandbox_job(&self, id: Uuid) -> Result<Option<SandboxJobRecord>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id, title, description, status, user_id, project_dir,
                       success, failure_reason, created_at, started_at, completed_at
                FROM agent_jobs WHERE id = ?1 AND source = 'sandbox'
                "#,
                params![id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(Some(SandboxJobRecord {
                id: get_text(&row, 0).parse().unwrap_or_default(),
                task: get_text(&row, 1),
                credential_grants_json: get_text(&row, 2),
                status: get_text(&row, 3),
                user_id: get_text(&row, 4),
                project_dir: get_text(&row, 5),
                success: get_opt_bool(&row, 6),
                failure_reason: get_opt_text(&row, 7),
                created_at: get_ts(&row, 8),
                started_at: get_opt_ts(&row, 9),
                completed_at: get_opt_ts(&row, 10),
            })),
            None => Ok(None),
        }
    }

    async fn list_sandbox_jobs(&self) -> Result<Vec<SandboxJobRecord>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id, title, description, status, user_id, project_dir,
                       success, failure_reason, created_at, started_at, completed_at
                FROM agent_jobs WHERE source = 'sandbox'
                ORDER BY created_at DESC
                "#,
                (),
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut jobs = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            jobs.push(SandboxJobRecord {
                id: get_text(&row, 0).parse().unwrap_or_default(),
                task: get_text(&row, 1),
                credential_grants_json: get_text(&row, 2),
                status: get_text(&row, 3),
                user_id: get_text(&row, 4),
                project_dir: get_text(&row, 5),
                success: get_opt_bool(&row, 6),
                failure_reason: get_opt_text(&row, 7),
                created_at: get_ts(&row, 8),
                started_at: get_opt_ts(&row, 9),
                completed_at: get_opt_ts(&row, 10),
            });
        }
        Ok(jobs)
    }

    async fn update_sandbox_job_status(
        &self,
        id: Uuid,
        status: &str,
        success: Option<bool>,
        message: Option<&str>,
        started_at: Option<DateTime<Utc>>,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
                UPDATE agent_jobs SET
                    status = ?2,
                    success = COALESCE(?3, success),
                    failure_reason = COALESCE(?4, failure_reason),
                    started_at = COALESCE(?5, started_at),
                    completed_at = COALESCE(?6, completed_at)
                WHERE id = ?1 AND source = 'sandbox'
                "#,
            params![
                id.to_string(),
                status,
                success.map(|b| b as i64),
                message,
                fmt_opt_ts(&started_at),
                fmt_opt_ts(&completed_at),
            ],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn cleanup_stale_sandbox_jobs(&self) -> Result<u64, DatabaseError> {
        let conn = self.connect().await?;
        let now = fmt_ts(&Utc::now());
        let count = conn
            .execute(
                r#"
                UPDATE agent_jobs SET
                    status = 'interrupted',
                    failure_reason = 'Process restarted',
                    completed_at = ?1
                WHERE source = 'sandbox' AND status IN ('running', 'creating')
                "#,
                params![now],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        if count > 0 {
            tracing::info!("Marked {} stale sandbox jobs as interrupted", count);
        }
        Ok(count)
    }

    async fn sandbox_job_summary(&self) -> Result<SandboxJobSummary, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT status, COUNT(*) as cnt FROM agent_jobs WHERE source = 'sandbox' GROUP BY status",
                (),
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut summary = SandboxJobSummary::default();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            let status = get_text(&row, 0);
            let count = get_i64(&row, 1) as usize;
            summary.total += count;
            match status.as_str() {
                "creating" => summary.creating += count,
                "running" => summary.running += count,
                "completed" => summary.completed += count,
                "failed" => summary.failed += count,
                "interrupted" => summary.interrupted += count,
                _ => {}
            }
        }
        Ok(summary)
    }

    async fn list_sandbox_jobs_for_user(
        &self,
        user_id: &str,
    ) -> Result<Vec<SandboxJobRecord>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id, title, description, status, user_id, project_dir,
                       success, failure_reason, created_at, started_at, completed_at
                FROM agent_jobs WHERE source = 'sandbox' AND user_id = ?1
                ORDER BY created_at DESC
                "#,
                libsql::params![user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut jobs = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            jobs.push(SandboxJobRecord {
                id: get_text(&row, 0).parse().unwrap_or_default(),
                task: get_text(&row, 1),
                credential_grants_json: get_text(&row, 2),
                status: get_text(&row, 3),
                user_id: get_text(&row, 4),
                project_dir: get_text(&row, 5),
                success: get_opt_bool(&row, 6),
                failure_reason: get_opt_text(&row, 7),
                created_at: get_ts(&row, 8),
                started_at: get_opt_ts(&row, 9),
                completed_at: get_opt_ts(&row, 10),
            });
        }
        Ok(jobs)
    }

    async fn list_all_jobs_for_user(
        &self,
        user_id: &str,
    ) -> Result<Vec<SandboxJobRecord>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id, title, description, status, user_id, project_dir,
                       success, failure_reason, created_at, started_at, completed_at
                FROM agent_jobs WHERE user_id = ?1
                ORDER BY created_at DESC
                "#,
                libsql::params![user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut jobs = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            jobs.push(SandboxJobRecord {
                id: get_text(&row, 0).parse().unwrap_or_default(),
                task: get_text(&row, 1),
                credential_grants_json: get_text(&row, 2),
                status: get_text(&row, 3),
                user_id: get_text(&row, 4),
                project_dir: get_text(&row, 5),
                success: get_opt_bool(&row, 6),
                failure_reason: get_opt_text(&row, 7),
                created_at: get_ts(&row, 8),
                started_at: get_opt_ts(&row, 9),
                completed_at: get_opt_ts(&row, 10),
            });
        }
        Ok(jobs)
    }

    async fn all_jobs_summary_for_user(
        &self,
        user_id: &str,
    ) -> Result<SandboxJobSummary, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT status, COUNT(*) as cnt FROM agent_jobs WHERE user_id = ?1 GROUP BY status",
                libsql::params![user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut summary = SandboxJobSummary::default();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            let status = get_text(&row, 0);
            let count = get_i64(&row, 1) as usize;
            summary.total += count;
            match status.as_str() {
                "creating" | "pending" => summary.creating += count,
                "running" | "in_progress" => summary.running += count,
                "completed" | "submitted" | "accepted" => summary.completed += count,
                "failed" | "cancelled" => summary.failed += count,
                "interrupted" => summary.interrupted += count,
                "stuck" => summary.running += count,
                _ => {}
            }
        }
        Ok(summary)
    }

    async fn sandbox_job_summary_for_user(
        &self,
        user_id: &str,
    ) -> Result<SandboxJobSummary, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT status, COUNT(*) as cnt FROM agent_jobs WHERE source = 'sandbox' AND user_id = ?1 GROUP BY status",
                libsql::params![user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut summary = SandboxJobSummary::default();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            let status = get_text(&row, 0);
            let count = get_i64(&row, 1) as usize;
            summary.total += count;
            match status.as_str() {
                "creating" => summary.creating += count,
                "running" => summary.running += count,
                "completed" => summary.completed += count,
                "failed" => summary.failed += count,
                "interrupted" => summary.interrupted += count,
                _ => {}
            }
        }
        Ok(summary)
    }

    async fn sandbox_job_belongs_to_user(
        &self,
        job_id: Uuid,
        user_id: &str,
    ) -> Result<bool, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT 1 FROM agent_jobs WHERE id = ?1 AND user_id = ?2 AND source = 'sandbox'",
                libsql::params![job_id.to_string(), user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        let found = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(found.is_some())
    }

    async fn update_sandbox_job_mode(&self, id: Uuid, mode: &str) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        conn.execute(
            "UPDATE agent_jobs SET job_mode = ?2 WHERE id = ?1",
            params![id.to_string(), mode],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn get_sandbox_job_mode(&self, id: Uuid) -> Result<Option<String>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT job_mode FROM agent_jobs WHERE id = ?1",
                params![id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(Some(get_text(&row, 0))),
            None => Ok(None),
        }
    }

    async fn save_job_event(
        &self,
        job_id: Uuid,
        event_type: &str,
        data: &serde_json::Value,
    ) -> Result<(), DatabaseError> {
        let conn = self.connect().await?;
        conn.execute(
            "INSERT INTO job_events (job_id, event_type, data) VALUES (?1, ?2, ?3)",
            params![job_id.to_string(), event_type, data.to_string()],
        )
        .await
        .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(())
    }

    async fn list_job_events(
        &self,
        job_id: Uuid,
        limit: Option<i64>,
    ) -> Result<Vec<JobEventRecord>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = if let Some(n) = limit {
            conn.query(
                r#"
                SELECT id, job_id, event_type, data, created_at
                FROM (
                    SELECT id, job_id, event_type, data, created_at
                    FROM job_events WHERE job_id = ?1
                    ORDER BY id DESC
                    LIMIT ?2
                )
                ORDER BY id ASC
                "#,
                params![job_id.to_string(), n],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        } else {
            conn.query(
                r#"
                SELECT id, job_id, event_type, data, created_at
                FROM job_events WHERE job_id = ?1 ORDER BY id ASC
                "#,
                params![job_id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        };

        let mut events = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            events.push(JobEventRecord {
                id: get_i64(&row, 0),
                job_id: get_text(&row, 1).parse().unwrap_or_default(),
                event_type: get_text(&row, 2),
                data: get_json(&row, 3),
                created_at: get_ts(&row, 4),
            });
        }
        Ok(events)
    }

    async fn get_any_job(&self, id: Uuid) -> Result<Option<SandboxJobRecord>, DatabaseError> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id, title, description, status, user_id, project_dir,
                       success, failure_reason, created_at, started_at, completed_at
                FROM agent_jobs WHERE id = ?1
                "#,
                params![id.to_string()],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        match rows
            .next()
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?
        {
            Some(row) => Ok(Some(SandboxJobRecord {
                id: get_text(&row, 0).parse().unwrap_or_default(),
                task: get_text(&row, 1),
                credential_grants_json: get_text(&row, 2),
                status: get_text(&row, 3),
                user_id: get_text(&row, 4),
                project_dir: get_text(&row, 5),
                success: get_opt_bool(&row, 6),
                failure_reason: get_opt_text(&row, 7),
                created_at: get_ts(&row, 8),
                started_at: get_opt_ts(&row, 9),
                completed_at: get_opt_ts(&row, 10),
            })),
            None => Ok(None),
        }
    }

    async fn delete_job(&self, id: Uuid, user_id: &str) -> Result<bool, DatabaseError> {
        let conn = self.connect().await?;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;

        let mut eligible_rows = match conn
            .query(
                r#"
                SELECT 1
                FROM agent_jobs
                WHERE id = ?1 AND user_id = ?2
                  AND status NOT IN ('running', 'creating', 'in_progress', 'pending')
                LIMIT 1
                "#,
                params![id.to_string(), user_id],
            )
            .await
        {
            Ok(rows) => rows,
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(DatabaseError::Query(e.to_string()));
            }
        };

        let is_eligible = match eligible_rows.next().await {
            Ok(row) => row.is_some(),
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(DatabaseError::Query(e.to_string()));
            }
        };

        if !is_eligible {
            let _ = conn.execute("ROLLBACK", ()).await;
            return Ok(false);
        }

        if let Err(e) = conn
            .execute(
                "DELETE FROM job_events WHERE job_id = ?1",
                params![id.to_string()],
            )
            .await
        {
            let _ = conn.execute("ROLLBACK", ()).await;
            return Err(DatabaseError::Query(e.to_string()));
        }

        let count = match conn
            .execute(
                r#"
                DELETE FROM agent_jobs
                WHERE id = ?1 AND user_id = ?2
                  AND status NOT IN ('running', 'creating', 'in_progress', 'pending')
                "#,
                params![id.to_string(), user_id],
            )
            .await
        {
            Ok(count) => count,
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(DatabaseError::Query(e.to_string()));
            }
        };

        if count == 0 {
            let _ = conn.execute("ROLLBACK", ()).await;
            return Ok(false);
        }

        conn.execute("COMMIT", ())
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(true)
    }

    async fn update_job_title(
        &self,
        id: Uuid,
        user_id: &str,
        title: &str,
    ) -> Result<bool, DatabaseError> {
        let conn = self.connect().await?;
        let count = conn
            .execute(
                "UPDATE agent_jobs SET title = ?1 WHERE id = ?2 AND user_id = ?3",
                params![title, id.to_string(), user_id],
            )
            .await
            .map_err(|e| DatabaseError::Query(e.to_string()))?;
        Ok(count > 0)
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use uuid::Uuid;

    use crate::db::libsql::LibSqlBackend;
    use crate::db::{Database, SandboxStore};
    use crate::history::SandboxJobRecord;

    async fn test_backend() -> (LibSqlBackend, tempfile::TempDir) {
        let tempdir = tempfile::tempdir().unwrap();
        let db_path = tempdir.path().join("sandbox-tests.db");
        let backend = LibSqlBackend::new_local(&db_path).await.unwrap();
        backend.run_migrations().await.unwrap();
        (backend, tempdir)
    }

    fn sandbox_job(id: Uuid, user_id: &str, status: &str) -> SandboxJobRecord {
        SandboxJobRecord {
            id,
            task: format!("task-{status}"),
            status: status.to_string(),
            user_id: user_id.to_string(),
            project_dir: "/tmp/work".to_string(),
            success: None,
            failure_reason: None,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            credential_grants_json: "[]".to_string(),
        }
    }

    #[tokio::test]
    async fn cleanup_stale_sandbox_jobs_only_updates_creating_and_running() {
        let (backend, _tempdir) = test_backend().await;

        let creating_id = Uuid::new_v4();
        let running_id = Uuid::new_v4();
        let completed_id = Uuid::new_v4();

        backend
            .save_sandbox_job(&sandbox_job(creating_id, "u1", "creating"))
            .await
            .unwrap();
        backend
            .save_sandbox_job(&sandbox_job(running_id, "u1", "running"))
            .await
            .unwrap();
        backend
            .save_sandbox_job(&sandbox_job(completed_id, "u1", "completed"))
            .await
            .unwrap();

        let conn = backend.connect().await.unwrap();
        let direct_id = Uuid::new_v4();
        conn.execute(
            "INSERT INTO agent_jobs (id, title, description, status, source, user_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            libsql::params![direct_id.to_string(), "direct", "direct", "running", "direct", "u1"],
        )
        .await
        .unwrap();

        let updated = backend.cleanup_stale_sandbox_jobs().await.unwrap();
        assert_eq!(updated, 2);

        let creating = backend.get_sandbox_job(creating_id).await.unwrap().unwrap();
        let running = backend.get_sandbox_job(running_id).await.unwrap().unwrap();
        let completed = backend
            .get_sandbox_job(completed_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(creating.status, "interrupted");
        assert_eq!(running.status, "interrupted");
        assert_eq!(completed.status, "completed");

        let mut rows = conn
            .query(
                "SELECT status FROM agent_jobs WHERE id = ?1",
                libsql::params![direct_id.to_string()],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let status: String = row.get(0).unwrap_or_default();
        assert_eq!(status, "running");
    }

    #[tokio::test]
    async fn list_job_events_limit_returns_latest_n_in_ascending_order() {
        let (backend, _tempdir) = test_backend().await;
        let job_id = Uuid::new_v4();
        backend
            .save_sandbox_job(&sandbox_job(job_id, "u1", "running"))
            .await
            .unwrap();

        backend
            .save_job_event(job_id, "created", &serde_json::json!({"n": 1}))
            .await
            .unwrap();
        backend
            .save_job_event(job_id, "heartbeat", &serde_json::json!({"n": 2}))
            .await
            .unwrap();
        backend
            .save_job_event(job_id, "done", &serde_json::json!({"n": 3}))
            .await
            .unwrap();

        let all = backend.list_job_events(job_id, None).await.unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(
            all.iter()
                .map(|e| e.event_type.as_str())
                .collect::<Vec<_>>(),
            vec!["created", "heartbeat", "done"]
        );

        let limited = backend.list_job_events(job_id, Some(2)).await.unwrap();
        assert_eq!(limited.len(), 2);
        assert_eq!(
            limited
                .iter()
                .map(|e| e.event_type.as_str())
                .collect::<Vec<_>>(),
            vec!["heartbeat", "done"]
        );
    }

    #[tokio::test]
    async fn delete_job_rejects_non_terminal_without_deleting_events() {
        let (backend, _tempdir) = test_backend().await;
        let job_id = Uuid::new_v4();
        backend
            .save_sandbox_job(&sandbox_job(job_id, "u1", "running"))
            .await
            .unwrap();
        backend
            .save_job_event(job_id, "progress", &serde_json::json!({"pct": 50}))
            .await
            .unwrap();

        let deleted = backend.delete_job(job_id, "u1").await.unwrap();
        assert!(!deleted);

        let job = backend.get_sandbox_job(job_id).await.unwrap();
        assert!(job.is_some());

        let events = backend.list_job_events(job_id, None).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "progress");
    }

    #[tokio::test]
    async fn all_jobs_summary_for_user_maps_both_agent_and_sandbox_statuses() {
        let (backend, _tempdir) = test_backend().await;
        let conn = backend.connect().await.unwrap();

        let statuses = [
            "pending",
            "creating",
            "in_progress",
            "running",
            "stuck",
            "submitted",
            "accepted",
            "completed",
            "cancelled",
            "failed",
            "interrupted",
        ];
        for status in statuses {
            conn.execute(
                "INSERT INTO agent_jobs (id, title, description, status, source, user_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                libsql::params![Uuid::new_v4().to_string(), status, "desc", status, "sandbox", "u1"],
            )
            .await
            .unwrap();
        }

        let summary = backend.all_jobs_summary_for_user("u1").await.unwrap();
        assert_eq!(summary.total, 11);
        assert_eq!(summary.creating, 2);
        assert_eq!(summary.running, 3);
        assert_eq!(summary.completed, 3);
        assert_eq!(summary.failed, 2);
        assert_eq!(summary.interrupted, 1);
    }
}
