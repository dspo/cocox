//! Wire types for the OpenAI Chat Completions API (`POST /v1/chat/completions`).
//!
//! Hand-maintained against the [OpenAI Chat Completions API
//! reference](https://platform.openai.com/docs/api-reference/chat/create).
//! Only the shapes Codex actually reads or writes are modeled — the request
//! side is `Serialize` (built by [`crate::chat_completions_mapping`]) and the
//! streaming side is `Deserialize` (consumed by [`crate::sse::chat_completions`]).
//!
//! Unlike the Responses API, Chat Completions streams *deltas*: tool calls and
//! text arrive incrementally and must be aggregated client-side. The chunk
//! types here therefore model a single SSE `data:` payload, not a full
//! response.

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;

// ── Request side ─────────────────────────────────────────────────────────────

/// Canonical input payload for the Chat Completions API (`POST /v1/chat/completions`).
///
/// Built by [`crate::chat_completions_mapping::map_to_chat_completions_request`]
/// from Codex's internal [`crate::common::ResponsesApiRequest`]. Only fields
/// Chat Completions understands are emitted; Responses-only concepts
/// (`store`, `include`, `prompt_cache_key`, `text`, `client_metadata`) are
/// dropped during mapping.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessageParam>,
    /// `max_completion_tokens` is the modern field; `max_tokens` remains the
    /// only accepted name on many OpenAI-compatible servers, so the mapper
    /// picks the field name based on provider capability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ChatCompletionTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Required to be `true` for streaming.
    pub stream: bool,
    /// Always sent with `include_usage: true` so the final chunk carries
    /// token usage; without it Codex cannot report usage.
    pub stream_options: ChatCompletionStreamOptions,
    /// Reasoning effort for o-series models exposed via Chat Completions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Optional `stop` sequences (mirrors the Responses API `stop` semantics).
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionStreamOptions {
    pub include_usage: bool,
}

/// A single conversation turn sent to the Chat Completions API.
///
/// Chat Completions uses a tagged `role` shape rather than the Responses
/// API's `type`-tagged items, so each variant maps to a distinct role.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatCompletionMessageParam {
    System { content: ChatCompletionContent },
    Developer { content: ChatCompletionContent },
    User { content: ChatCompletionContent },
    /// An assistant turn. `content` may be `null` when the turn only carries
    /// `tool_calls`; `tool_calls` is absent for plain text turns.
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ChatCompletionContent>,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        tool_calls: Vec<ChatCompletionAssistantToolCall>,
    },
    /// The result of a function/tool call, returned to the model.
    Tool {
        tool_call_id: String,
        content: String,
    },
}

/// Message content — either a plain string (the common case) or a list of
/// typed parts (used for multimodal input).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ChatCompletionContent {
    Text(String),
    Parts(Vec<ChatCompletionContentPart>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionContentPart {
    Text { text: String },
    ImageUrl { image_url: ChatCompletionImageUrl },
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// An assistant-emitted tool call on the request side (replayed history).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionAssistantToolCall {
    /// Stable index used by Chat Completions to align streaming deltas.
    pub index: i64,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatCompletionAssistantToolCallFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionAssistantToolCallFunction {
    pub name: String,
    /// JSON-encoded argument string (Chat Completions uses a string, not an object).
    pub arguments: String,
}

/// Tool definition in Chat Completions `function`-calling shape.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatCompletionToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema describing the arguments.
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Tool choice control. Mirrors the Chat Completions `tool_choice` field.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ChatCompletionToolChoice {
    /// `"auto"` | `"none"` | `"required"`.
    Mode(String),
    Named(ChatCompletionNamedToolChoice),
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionNamedToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatCompletionNamedToolChoiceFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionNamedToolChoiceFunction {
    pub name: String,
}

// ── Streaming (response) side ────────────────────────────────────────────────

/// A single SSE `data:` payload from the Chat Completions streaming endpoint.
///
/// OpenAI emits one chunk per token (or per tool-call argument fragment); the
/// final chunk carries `usage` when `stream_options.include_usage` is set, and
/// the stream terminates with the literal `data: [DONE]` line (handled outside
/// the deserializer).
#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: Option<String>,
    /// The model the server actually used; surfaced via
    /// [`crate::common::ResponseEvent::ServerModel`] on the first chunk that
    /// carries it.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<ChatCompletionChoice>,
    /// Present on the final chunk when `stream_options.include_usage = true`.
    #[serde(default)]
    pub usage: Option<ChatCompletionUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChoice {
    pub delta: ChatCompletionDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Incremental delta for a choice. All fields are optional because each chunk
/// only carries the newly produced fragment.
#[derive(Debug, Default, Deserialize)]
pub struct ChatCompletionDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ChatCompletionToolCallDelta>,
}

/// A streaming tool-call fragment. `index` aligns fragments across chunks;
/// `id` and `function.name` appear only on the first fragment for a given
/// `index`, while `function.arguments` is streamed incrementally.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionToolCallDelta {
    #[serde(default)]
    pub index: i64,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChatCompletionToolCallFunctionDelta>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct ChatCompletionToolCallFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Token usage emitted on the final chunk.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatCompletionUsage {
    #[serde(default)]
    pub prompt_tokens: i64,
    #[serde(default)]
    pub completion_tokens: i64,
    #[serde(default)]
    pub total_tokens: i64,
    #[serde(default)]
    pub prompt_tokens_details: Option<ChatCompletionPromptTokensDetails>,
    #[serde(default)]
    pub completion_tokens_details: Option<ChatCompletionCompletionTokensDetails>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatCompletionPromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: i64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatCompletionCompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: i64,
}

/// The literal `data: [DONE]` sentinel terminating a Chat Completions stream.
pub const STREAM_DONE_SENTINEL: &str = "[DONE]";

/// Maps a Chat Completions `finish_reason` to the closest Codex stop semantics.
///
/// Returns `Some(true)` when the model naturally ended its turn, `Some(false)`
/// when it stopped to emit tool calls (or hit a limit), and `None` when
/// streaming is still in progress (`finish_reason` is absent.
pub fn finish_reason_ends_turn(finish_reason: &str) -> Option<bool> {
    match finish_reason {
        "stop" | "content_filter" => Some(true),
        "tool_calls" | "function_call" => Some(false),
        "length" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serializes_minimal_request() {
        let req = ChatCompletionsRequest {
            model: "gpt-4o".into(),
            messages: vec![ChatCompletionMessageParam::User {
                content: ChatCompletionContent::Text("hello".into()),
            }],
            max_tokens: Some(128),
            max_completion_tokens: None,
            temperature: None,
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            stream: true,
            stream_options: ChatCompletionStreamOptions { include_usage: true },
            reasoning_effort: None,
            stop: Vec::new(),
        };

        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["model"], "gpt-4o");
        assert_eq!(value["stream"], true);
        assert_eq!(value["stream_options"]["include_usage"], true);
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][0]["content"], "hello");
        // Empty tools / stop must be omitted.
        assert!(value.get("tools").is_none());
        assert!(value.get("stop").is_none());
        assert!(value.get("temperature").is_none());
    }

    #[test]
    fn serializes_assistant_tool_call_history() {
        let msg = ChatCompletionMessageParam::Assistant {
            content: None,
            tool_calls: vec![ChatCompletionAssistantToolCall {
                index: 0,
                id: "call_1".into(),
                kind: "function".into(),
                function: ChatCompletionAssistantToolCallFunction {
                    name: "search".into(),
                    arguments: "{\"q\":\"rust\"}".into(),
                },
            }],
        };
        let value = serde_json::to_value(&msg).unwrap();
        assert_eq!(value["role"], "assistant");
        assert!(value.get("content").is_none());
        assert_eq!(value["tool_calls"][0]["index"], 0);
        assert_eq!(value["tool_calls"][0]["id"], "call_1");
        assert_eq!(value["tool_calls"][0]["type"], "function");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "search");
        assert_eq!(
            value["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn deserializes_text_delta_chunk() {
        let payload = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "Hello" },
                "finish_reason": null
            }]
        });
        let chunk: ChatCompletionChunk = serde_json::from_value(payload).unwrap();
        assert_eq!(chunk.id.as_deref(), Some("chatcmpl-1"));
        let choice = chunk.choices.first().unwrap();
        assert_eq!(choice.delta.role.as_deref(), Some("assistant"));
        assert_eq!(choice.delta.content.as_deref(), Some("Hello"));
        assert!(choice.finish_reason.is_none());
    }

    #[test]
    fn deserializes_tool_call_delta_chunk() {
        let payload = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "search", "arguments": "{\"q\":" }
                    }]
                },
                "finish_reason": null
            }]
        });
        let chunk: ChatCompletionChunk = serde_json::from_value(payload).unwrap();
        let tc = chunk.choices[0].delta.tool_calls[0].clone();
        assert_eq!(tc.index, 0);
        assert_eq!(tc.id.as_deref(), Some("call_1"));
        assert_eq!(tc.function.unwrap().arguments.unwrap(), "{\"q\":");
    }

    #[test]
    fn deserializes_final_usage_chunk() {
        let payload = json!({
            "id": "chatcmpl-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": { "cached_tokens": 4 },
                "completion_tokens_details": { "reasoning_tokens": 0 }
            }
        });
        let chunk: ChatCompletionChunk = serde_json::from_value(payload).unwrap();
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
        assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, 4);
    }

    #[test]
    fn finish_reason_mapping() {
        assert_eq!(finish_reason_ends_turn("stop"), Some(true));
        assert_eq!(finish_reason_ends_turn("tool_calls"), Some(false));
        assert_eq!(finish_reason_ends_turn("length"), Some(false));
        assert_eq!(finish_reason_ends_turn("content_filter"), Some(true));
        assert_eq!(finish_reason_ends_turn(""), None);
    }
}
