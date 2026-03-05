# K2.5 HTTP 500 Debug Investigation

## Problem
IronClaw gets HTTP 500 from Baseten (and Fireworks) when sending tool-calling requests to Kimi K2.5. The same endpoint works fine via curl.

## What works
- **curl → Baseten K2.5**: 14 tools, 42KB payload, array content format, tool_choice, enums — all fine, 200 OK
- **IronClaw → OpenRouter K2.5**: works (OpenRouter normalizes requests)
- **IronClaw → Baseten/Fireworks GLM-5**: works (via RigAdapter with lenient schemas)

## What fails
- **IronClaw → Baseten K2.5**: HTTP 500, both via:
  - `NearAiChatProvider` (our own simple HTTP client)
  - `RigAdapter` (rig-core's OpenAI CompletionsClient)

## Key finding
Both IronClaw code paths fail, but curl to the same endpoint with the same tools succeeds. This means something in the **serialized request body** that IronClaw/rig-core produces is different from what curl sends, and Baseten's vLLM/SGLang rejects it.

## What we've ruled out
| Hypothesis | Test | Result |
|---|---|---|
| Payload too large | Reduced from 233KB→44KB→31KB | Still 500 |
| Too many tools | Tested 14 tools via curl | 200 OK |
| `tool_choice: "auto"` | Tested with and without | 200 OK |
| `enum` in schemas | Tested with enums via curl | 200 OK |
| Array content format (`[{"type":"text","text":"..."}]`) | Tested via curl | 200 OK |
| Array content + tools | Tested via curl | 200 OK |
| Strict schemas (`additionalProperties: false`) | K2.5 uses lenient path | Still 500 |

## Current blocker
We cannot see the **exact HTTP request body** that rig-core sends. Rig-core has TRACE-level logging (`target: "rig::completions"`) but it doesn't fire even with `RUST_LOG=rig_core=trace,rig=trace`. The `enabled!(Level::TRACE)` guard in rig-core (line 1176 of `completion/mod.rs`) isn't passing.

## What needs to happen
1. **Capture the exact request body** rig-core sends to Baseten
2. **Diff it** against a working curl request
3. **Identify the field/format** causing the 500

### Options to capture the request
- **Option A**: Add debug serialization in `RigAdapter::complete_with_tools` — serialize the `CompletionRequest` ourselves before rig sends it (requires importing rig's internal types or reconstructing the request)
- **Option B**: Use a request interceptor / HTTP proxy to capture the raw request
- **Option C**: Bypass rig-core entirely — use `NearAiChatProvider`'s simple reqwest-based HTTP client with the same lenient schema normalization that RigAdapter applies. This gives us full control over serialization AND debug logging.
- **Option D**: Write a standalone test that constructs the same `CompletionRequest` rig-core would build, serializes it, and dumps the JSON for manual curl testing

### Likely root cause candidates
Since curl works but rig-core doesn't, the difference is likely:
1. **rig-core serializes `content` as `[{"type":"text","text":"..."}]` array** — BUT curl test shows Baseten accepts this
2. **rig-core adds extra fields** via `#[serde(flatten)] additional_params` — could inject unexpected keys
3. **rig-core serializes `tool_choice` as an object** (`{"type":"auto"}`) instead of string (`"auto"`) — rig-core uses a `ToolChoice` enum, not a plain string
4. **rig-core omits `max_tokens`** or formats it differently
5. **Something in rig-core's `Message` enum serialization** that differs from plain JSON

## Files involved
- `src/llm/rig_adapter.rs` — IronClaw's rig-core wrapper
- `src/llm/mod.rs` — Provider routing (K2.5 special-casing removed, now goes through RigAdapter)
- `src/llm/nearai_chat.rs` — Alternative simple HTTP client (works for NEAR AI, was tried for K2.5)
- `~/.cargo/registry/src/.../rig-core-0.30.0/src/providers/openai/completion/mod.rs` — rig-core's request construction
