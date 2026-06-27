//! Wire types for the Anthropic Messages API.
//!
//! Hand-maintained against the [Anthropic Messages API
//! reference](https://docs.anthropic.com/en/api/messages). Only the shapes
//! Codex actually reads or writes are modeled.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Required `anthropic-version` header value for the Messages API.
/// Covers text + tool use + streaming; bump when adopting newer features
/// (e.g. extended thinking requires a later version).
pub const ANTHROPIC_API_VERSION: &str = "2023-06-01";


/// Canonical input payload for the Anthropic Messages API (POST /v1/messages).
///
/// Fields mirror the [Anthropic Messages API reference](https://docs.anthropic.com/en/api/messages)
/// but only include what Codex actually needs to send.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessageParam>,
    pub max_tokens: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystemPrompt>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
}

/// System prompt, either a plain string or an array of text blocks.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum AnthropicSystemPrompt {
    Text(String),
    Blocks(Vec<AnthropicTextBlockParam>),
}

/// A single conversation turn sent to the Messages API.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMessageParam {
    pub role: String,
    pub content: AnthropicMessageContent,
}

/// Per-message content — either a simple string (convenience) or a list of blocks.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlockParam>),
}

/// Content blocks accepted on the request side.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockParam {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        source: AnthropicImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: AnthropicToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicImageSource {
    Url { url: String },
    Base64 {
        media_type: String,
        data: String,
    },
}

/// Content of a tool result — either a plain string or structured blocks.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum AnthropicToolResultContent {
    Text(String),
    Blocks(Vec<AnthropicToolResultBlock>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolResultBlock {
    Text { text: String },
    Image { source: AnthropicImageSource },
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicCacheControl {
    #[serde(rename = "type")]
    pub cache_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicTextBlockParam {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

/// Tool definition for the Messages API.
///
/// Follows the Anthropic tool schema: `{ name, description?, input_schema }`.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Tool choice control.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    None,
}

/// Extended thinking configuration.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicThinkingConfig {
    Enabled {
        budget_tokens: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<String>,
    },
    Disabled,
    Adaptive {
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

// ── Anthropic SSE stream event types ─────────────────────────────────────────

/// A single SSE event from the Anthropic Messages streaming endpoint.
///
/// Deserialized from `data: {...}` lines inside the SSE stream.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart {
        message: AnthropicResponseMessage,
    },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: i64,
        content_block: AnthropicResponseContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        index: i64,
        delta: AnthropicContentBlockDelta,
    },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        index: i64,
    },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: AnthropicUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
}

/// The `message` object carried by `message_start`.
#[derive(Debug, Deserialize)]
pub struct AnthropicResponseMessage {
    pub id: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

/// Token usage (emitted by `message_start` and `message_delta`).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: Option<i64>,
    #[serde(default)]
    pub output_tokens: Option<i64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<i64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<i64>,
}

/// A content block inside a `content_block_start` event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicResponseContentBlock {
    Text { text: String },
    Thinking { thinking: String, signature: Option<String> },
    RedactedThinking { data: String },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Option<Value>,
    },
}

/// Delta payload inside a `content_block_delta` event.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

/// The `delta` payload inside a `message_delta` event.
#[derive(Debug, Deserialize)]
pub struct AnthropicMessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
}
