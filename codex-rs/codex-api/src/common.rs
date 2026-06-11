use crate::error::ApiError;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::config_types::Verbosity as VerbosityConfig;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::protocol::ModelVerification;
use codex_protocol::protocol::RateLimitSnapshot;
use codex_protocol::protocol::TokenUsage;
use codex_protocol::protocol::TurnModerationMetadataEvent;
use codex_protocol::protocol::W3cTraceContext;
use futures::Stream;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use tokio::sync::mpsc;

pub const WS_REQUEST_HEADER_TRACEPARENT_CLIENT_METADATA_KEY: &str = "ws_request_header_traceparent";
pub const WS_REQUEST_HEADER_TRACESTATE_CLIENT_METADATA_KEY: &str = "ws_request_header_tracestate";

/// Canonical input payload for the compaction endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct CompactionInput<'a> {
    pub model: &'a str,
    pub input: &'a [ResponseItem],
    #[serde(skip_serializing_if = "str::is_empty")]
    pub instructions: &'a str,
    pub tools: Vec<Value>,
    pub parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextControls>,
}

/// Canonical input payload for the memory summarize endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct MemorySummarizeInput {
    pub model: String,
    #[serde(rename = "traces")]
    pub raw_memories: Vec<RawMemory>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawMemory {
    pub id: String,
    pub metadata: RawMemoryMetadata,
    pub items: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawMemoryMetadata {
    pub source_path: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct MemorySummarizeOutput {
    #[serde(rename = "trace_summary", alias = "raw_memory")]
    pub raw_memory: String,
    pub memory_summary: String,
}

#[derive(Debug)]
pub enum ResponseEvent {
    Created,
    OutputItemDone(ResponseItem),
    OutputItemAdded(ResponseItem),
    /// Emitted when the server includes `OpenAI-Model` on the stream response.
    /// This can differ from the requested model when backend safety routing applies.
    ServerModel(String),
    /// Emitted when the server recommends additional account verification.
    ModelVerifications(Vec<ModelVerification>),
    /// Emitted when the server includes moderation metadata for first-party turn presentation.
    TurnModerationMetadata(TurnModerationMetadataEvent),
    /// Emitted when `X-Reasoning-Included: true` is present on the response,
    /// meaning the server already accounted for past reasoning tokens and the
    /// client should not re-estimate them.
    ServerReasoningIncluded(bool),
    Completed {
        response_id: String,
        token_usage: Option<TokenUsage>,
        /// Did the model affirmatively end its turn? Some providers do not set this,
        /// so we rely on fallback logic when this is `None`.
        end_turn: Option<bool>,
    },
    OutputTextDelta(String),
    ToolCallInputDelta {
        item_id: String,
        call_id: Option<String>,
        delta: String,
    },
    ReasoningSummaryDelta {
        delta: String,
        summary_index: i64,
    },
    ReasoningContentDelta {
        delta: String,
        content_index: i64,
    },
    ReasoningSummaryPartAdded {
        summary_index: i64,
    },
    RateLimits(RateLimitSnapshot),
    ModelsEtag(String),
}

#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningContext {
    Auto,
    CurrentTurn,
    AllTurns,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct Reasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffortConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummaryConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ReasoningContext>,
}

#[derive(Debug, Serialize, Default, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TextFormatType {
    #[default]
    JsonSchema,
}

#[derive(Debug, Serialize, Default, Clone, PartialEq)]
pub struct TextFormat {
    /// Format type used by the OpenAI text controls.
    pub r#type: TextFormatType,
    /// When true, the server is expected to strictly validate responses.
    pub strict: bool,
    /// JSON schema for the desired output.
    pub schema: Value,
    /// Friendly name for the format, used in telemetry/debugging.
    pub name: String,
}

/// Controls the `text` field for the Responses API, combining verbosity and
/// optional JSON schema output formatting.
#[derive(Debug, Serialize, Default, Clone, PartialEq)]
pub struct TextControls {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<OpenAiVerbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

#[derive(Debug, Serialize, Default, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiVerbosity {
    Low,
    #[default]
    Medium,
    High,
}

impl From<VerbosityConfig> for OpenAiVerbosity {
    fn from(v: VerbosityConfig) -> Self {
        match v {
            VerbosityConfig::Low => OpenAiVerbosity::Low,
            VerbosityConfig::Medium => OpenAiVerbosity::Medium,
            VerbosityConfig::High => OpenAiVerbosity::High,
        }
    }
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct ResponsesApiRequest {
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub instructions: String,
    pub input: Vec<ResponseItem>,
    pub tools: Vec<serde_json::Value>,
    pub tool_choice: String,
    pub parallel_tool_calls: bool,
    pub reasoning: Option<Reasoning>,
    pub store: bool,
    pub stream: bool,
    pub include: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextControls>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_metadata: Option<HashMap<String, String>>,
}

impl From<&ResponsesApiRequest> for ResponseCreateWsRequest {
    fn from(request: &ResponsesApiRequest) -> Self {
        Self {
            model: request.model.clone(),
            instructions: request.instructions.clone(),
            previous_response_id: None,
            input: request.input.clone(),
            tools: request.tools.clone(),
            tool_choice: request.tool_choice.clone(),
            parallel_tool_calls: request.parallel_tool_calls,
            reasoning: request.reasoning.clone(),
            store: request.store,
            stream: request.stream,
            include: request.include.clone(),
            service_tier: request.service_tier.clone(),
            prompt_cache_key: request.prompt_cache_key.clone(),
            text: request.text.clone(),
            generate: None,
            client_metadata: request.client_metadata.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ResponseCreateWsRequest {
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub instructions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    pub input: Vec<ResponseItem>,
    pub tools: Vec<Value>,
    pub tool_choice: String,
    pub parallel_tool_calls: bool,
    pub reasoning: Option<Reasoning>,
    pub store: bool,
    pub stream: bool,
    pub include: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextControls>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_metadata: Option<HashMap<String, String>>,
}

pub fn response_create_client_metadata(
    client_metadata: Option<HashMap<String, String>>,
    trace: Option<&W3cTraceContext>,
) -> Option<HashMap<String, String>> {
    let mut client_metadata = client_metadata.unwrap_or_default();

    if let Some(traceparent) = trace.and_then(|trace| trace.traceparent.as_deref()) {
        client_metadata.insert(
            WS_REQUEST_HEADER_TRACEPARENT_CLIENT_METADATA_KEY.to_string(),
            traceparent.to_string(),
        );
    }
    if let Some(tracestate) = trace.and_then(|trace| trace.tracestate.as_deref()) {
        client_metadata.insert(
            WS_REQUEST_HEADER_TRACESTATE_CLIENT_METADATA_KEY.to_string(),
            tracestate.to_string(),
        );
    }

    (!client_metadata.is_empty()).then_some(client_metadata)
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
#[allow(clippy::large_enum_variant)]
pub enum ResponsesWsRequest {
    #[serde(rename = "response.create")]
    ResponseCreate(ResponseCreateWsRequest),
}

pub fn create_text_param_for_request(
    verbosity: Option<VerbosityConfig>,
    output_schema: &Option<Value>,
    output_schema_strict: bool,
) -> Option<TextControls> {
    if verbosity.is_none() && output_schema.is_none() {
        return None;
    }

    Some(TextControls {
        verbosity: verbosity.map(std::convert::Into::into),
        format: output_schema.as_ref().map(|schema| TextFormat {
            r#type: TextFormatType::JsonSchema,
            strict: output_schema_strict,
            schema: schema.clone(),
            name: "codex_output_schema".to_string(),
        }),
    })
}

pub struct ResponseStream {
    pub rx_event: mpsc::Receiver<Result<ResponseEvent, ApiError>>,
    /// Server-assigned `x-request-id` response header, when present.
    pub upstream_request_id: Option<String>,
}

impl Stream for ResponseStream {
    type Item = Result<ResponseEvent, ApiError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx_event.poll_recv(cx)
    }
}

// ── Anthropic Messages API types ─────────────────────────────────────────────

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
    #[serde(flatten)]
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
