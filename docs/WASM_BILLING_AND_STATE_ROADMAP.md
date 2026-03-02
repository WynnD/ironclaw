# WASM Billing and Durable State Roadmap

This document captures the P3 roadmap items requested for WASM tools.

## 1) Capability-Based Billing (`cost_per_call`)

Goal: allow operators to charge/account for third-party WASM tool usage using capabilities metadata.

Proposed shape:
- Add optional `cost_per_call` to `CapabilitiesFile` (schema) and persist it in `tool_capabilities` storage.
- On each WASM tool execution, record a billing event in DB with:
  - `tool_name`, `user_id`, `job_id`, `timestamp`, `cost_per_call`, `execution_duration_ms`.
- Keep billing orthogonal to rate limits and safety checks.

Design constraints:
- Backward compatible: missing `cost_per_call` means no billing.
- Deterministic accounting: one charge per successful execution entrypoint.
- Multi-backend parity: PostgreSQL and libSQL schema support.

## 2) Durable WASM Tool State (Host KV)

Goal: provide a safe host-provided key-value state store for WASM tools across invocations.

Proposed host API (conceptual):
- `state_get(key) -> Option<bytes>`
- `state_set(key, value)`
- `state_delete(key)`
- Optional `state_list(prefix)` for namespaced scans.

Security and isolation:
- Namespace by `(user_id, tool_name)` to prevent cross-tool and cross-user leakage.
- Enforce per-tool quotas (total bytes, max value size, write rate).
- Validate keys and prohibit path-like traversal semantics.

Storage model:
- Persist in DB (`wasm_tool_state` table) with backend parity.
- Use upsert semantics and updated-at metadata.
- Consider optional TTL support in a later phase.

## Rollout Sequence

1. Add schema fields/tables and storage methods (both DB backends).
2. Extend capabilities parsing and runtime plumbing.
3. Add host functions and wrapper/runtime integration.
4. Add accounting/state observability in logs and admin tooling.
5. Add tests for quotas, isolation, and migration compatibility.
