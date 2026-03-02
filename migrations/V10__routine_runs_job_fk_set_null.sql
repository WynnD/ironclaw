-- Keep routine run history when deleting terminal jobs.
-- Existing behavior blocks job deletes due to routine_runs.job_id FK.
ALTER TABLE routine_runs
    DROP CONSTRAINT IF EXISTS routine_runs_job_id_fkey;

ALTER TABLE routine_runs
    ADD CONSTRAINT routine_runs_job_id_fkey
    FOREIGN KEY (job_id)
    REFERENCES agent_jobs(id)
    ON DELETE SET NULL;
