//! SSE parser for the Anthropic Messages API streaming endpoint.
//!
//! Converts the Anthropic SSE event stream into Codex's internal
//! `ResponseEvent` stream, maintaining a state machine that tracks
//! content block boundaries and tool-call aggregation.

use crate::anthropic::AnthropicContentBlockDelta;
use crate::anthropic::AnthropicResponseContentBlock;
use crate::anthropic::AnthropicStreamEvent;
use crate::anthropic::AnthropicUsage;
use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

/// State tracked across a single streaming turn for the Messages API.
#[derive(Debug, Default)]
struct MessagesStreamState {
    /// The `message.id` from the most recent `message_start`.
    response_id: Option<String>,
    /// Model slug from `message_start`.
    model: Option<String>,
    /// Accumulated usage from `message_start` and `message_delta`.
    usage: Option<AnthropicUsage>,
    /// The stop reason from `message_delta`.
    stop_reason: Option<String>,
    /// The tool-call id currently being assembled, keyed by content-block index.
    active_tool_call: Option<ToolCallState>,
    /// Accumulated text for the current text block.
    current_text: String,
    /// Have we seen at least one non-ping event?
    saw_non_ping: bool,
}

#[derive(Debug)]
struct ToolCallState {
    index: i64,
    call_id: String,
    name: String,
    /// Accumulated partial JSON fragments for the current tool call.
    partial_json: String,
}

pub fn spawn_messages_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
    turn_state: Option<Arc<OnceLock<String>>>,
) -> ResponseStream {
    let upstream_request_id = stream_response
        .headers
        .get("request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);

    if let Some(turn_state) = turn_state.as_ref()
        && let Some(header_value) = stream_response
            .headers
            .get("x-codex-turn-state")
            .and_then(|v| v.to_str().ok())
    {
        let _ = turn_state.set(header_value.to_string());
    }

    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);

    tokio::spawn(async move {
        process_messages_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

/// Drives the SSE event loop for the Messages API, translating Anthropic events
/// into `ResponseEvent`s.
pub async fn process_messages_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = MessagesStreamState::default();
    let mut response_error: Option<ApiError> = None;

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }

        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Messages SSE error: {e:#}");
                let _ = tx_event
                    .send(Err(ApiError::Stream(e.to_string())))
                    .await;
                return;
            }
            Ok(None) => {
                let error = response_error.unwrap_or(ApiError::Stream(
                    "messages stream closed before message_stop".into(),
                ));
                let _ = tx_event.send(Err(error)).await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "idle timeout waiting for Messages SSE".into(),
                    )))
                    .await;
                return;
            }
        };

        trace!("Messages SSE event: {}", &sse.data);

        let event: AnthropicStreamEvent = match serde_json::from_str(&sse.data) {
            Ok(event) => event,
            Err(e) => {
                debug!(
                    "Failed to parse Messages SSE event: {e}, data: {}",
                    &sse.data
                );
                continue;
            }
        };

        match process_messages_event(event, &mut state) {
            Ok(Some(response_event)) => {
                let is_completed = matches!(response_event, ResponseEvent::Completed { .. });
                if tx_event.send(Ok(response_event)).await.is_err() {
                    return;
                }
                if is_completed {
                    return;
                }
            }
            Ok(None) => {}
            Err(err) => {
                response_error = Some(err);
            }
        };
    }
}

/// Processes a single Messages API SSE event using the accumulated `state`.
fn process_messages_event(
    event: AnthropicStreamEvent,
    state: &mut MessagesStreamState,
) -> Result<Option<ResponseEvent>, ApiError> {
    match event {
        AnthropicStreamEvent::Ping => {
            // Ping can appear before message_start; silently skip.
            return Ok(None);
        }

        AnthropicStreamEvent::MessageStart { message } => {
            if state.saw_non_ping {
                debug!("duplicate message_start in Messages stream");
            }
            state.saw_non_ping = true;
            state.response_id = Some(message.id);

            if let Some(model) = message.model {
                state.model = Some(model);
            }
            state.usage = message.usage;

            // Emit ServerModel if we have one.
            if let Some(ref model) = state.model {
                return Ok(Some(ResponseEvent::ServerModel(model.clone())));
            }
            return Ok(None);
        }

        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            state.saw_non_ping = true;
            match content_block {
                AnthropicResponseContentBlock::Text { text } => {
                    state.current_text = text;
                    return Ok(None);
                }
                AnthropicResponseContentBlock::ToolUse {
                    id,
                    name,
                    input: _,
                } => {
                    // Emit OutputItemAdded for the new tool call.
                    state.active_tool_call = Some(ToolCallState {
                        index,
                        call_id: id.clone(),
                        name: name.clone(),
                        partial_json: String::new(),
                    });

                    return Ok(Some(ResponseEvent::OutputItemAdded(
                        ResponseItem::FunctionCall {
                            id: None,
                            name,
                            namespace: None,
                            arguments: String::new(),
                            call_id: id,
                            internal_chat_message_metadata_passthrough: None,
                        },
                    )));
                }
                AnthropicResponseContentBlock::Thinking { .. }
                | AnthropicResponseContentBlock::RedactedThinking { .. } => {
                    // Thinking blocks are ignored for now (could be mapped to
                    // Reasoning events in the future).
                    return Ok(None);
                }
            }
        }

        AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
            state.saw_non_ping = true;
            match delta {
                AnthropicContentBlockDelta::TextDelta { text } => {
                    state.current_text.push_str(&text);
                    return Ok(Some(ResponseEvent::OutputTextDelta(text)));
                }
                AnthropicContentBlockDelta::InputJsonDelta { partial_json } => {
                    if let Some(ref mut tc) = state.active_tool_call
                        && tc.index == index
                    {
                        tc.partial_json.push_str(&partial_json);
                        return Ok(Some(ResponseEvent::ToolCallInputDelta {
                            item_id: tc.call_id.clone(),
                            call_id: Some(tc.call_id.clone()),
                            delta: partial_json,
                        }));
                    }
                    return Ok(None);
                }
                AnthropicContentBlockDelta::ThinkingDelta { .. }
                | AnthropicContentBlockDelta::SignatureDelta { .. } => {
                    return Ok(None);
                }
            }
        }

        AnthropicStreamEvent::ContentBlockStop { index } => {
            state.saw_non_ping = true;

            // If we were tracking a text block, emit the complete OutputItemDone.
            if state.active_tool_call.is_none() && !state.current_text.is_empty() {
                let text = std::mem::take(&mut state.current_text);
                return Ok(Some(ResponseEvent::OutputItemDone(ResponseItem::Message {
                    id: None,
                    role: "assistant".to_string(),
                    content: vec![ContentItem::OutputText { text }],
                    phase: None,
                    internal_chat_message_metadata_passthrough: None,
                })));
            }

            // If we were tracking a tool call, emit the complete FunctionCall.
            if let Some(tc) = state.active_tool_call.take()
                && tc.index == index
            {
                // Try to parse the accumulated partial JSON as the arguments.
                let arguments = tc.partial_json.clone();
                return Ok(Some(ResponseEvent::OutputItemDone(
                    ResponseItem::FunctionCall {
                        id: None,
                        name: tc.name,
                        namespace: None,
                        arguments,
                        call_id: tc.call_id,
                        internal_chat_message_metadata_passthrough: None,
                    },
                )));
            }

            return Ok(None);
        }

        AnthropicStreamEvent::MessageDelta { delta, usage } => {
            state.saw_non_ping = true;
            state.stop_reason = delta.stop_reason;
            state.usage = Some(usage);
            return Ok(None);
        }

        AnthropicStreamEvent::MessageStop => {
            state.saw_non_ping = true;
            let response_id = state.response_id.take().unwrap_or_default();
            let token_usage = state.usage.take().map(|u| TokenUsage {
                input_tokens: u.input_tokens.unwrap_or(0),
                cached_input_tokens: u.cache_read_input_tokens.unwrap_or(0),
                output_tokens: u.output_tokens.unwrap_or(0),
                reasoning_output_tokens: 0,
                total_tokens: u.input_tokens.unwrap_or(0) + u.output_tokens.unwrap_or(0),
            });

            // Determine end_turn from stop_reason.
            let end_turn = state
                .stop_reason
                .as_deref()
                .map(|reason| reason == "end_turn" || reason == "stop_sequence");

            return Ok(Some(ResponseEvent::Completed {
                response_id,
                token_usage,
                end_turn,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use bytes::Bytes;
    use codex_client::TransportError;
    use futures::TryStreamExt;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn idle_timeout() -> Duration {
        Duration::from_millis(1000)
    }

    async fn collect_messages_events(
        sse_json_events: Vec<serde_json::Value>,
    ) -> Vec<ResponseEvent> {
        let mut body = String::new();
        for e in sse_json_events {
            let kind = e
                .get("type")
                .and_then(|v| v.as_str())
                .expect("fixture missing type");
            body.push_str(&format!(
                "event: {kind}\ndata: {e}\n\n",
                kind = kind,
                e = serde_json::to_string(&e).unwrap()
            ));
        }

        let stream = futures::stream::iter(vec![Ok(Bytes::from(body))]);
        let stream: ByteStream = Box::pin(
            stream.map_err(|e: std::io::Error| TransportError::Network(e.to_string())),
        );

        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(8);
        tokio::spawn(process_messages_sse(
            stream,
            tx,
            idle_timeout(),
            /*telemetry*/ None,
        ));

        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev.expect("channel closed early"));
        }
        out
    }

    #[tokio::test]
    async fn parses_simple_text_turn() {
        let events = collect_messages_events(vec![
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_001",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": { "input_tokens": 100, "output_tokens": 0 }
                }
            }),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "Hello" }
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": " world" }
            }),
            json!({
                "type": "content_block_stop",
                "index": 0
            }),
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 5 }
            }),
            json!({
                "type": "message_stop"
            }),
        ])
        .await;

        // events: ServerModel, OutputTextDelta("Hello"), OutputTextDelta(" world"),
        //         OutputItemDone(Message), Completed
        assert!(events.len() >= 4, "got {} events: {events:?}", events.len());

        assert_matches!(&events[0], ResponseEvent::ServerModel(model) => {
            assert_eq!(model, "claude-sonnet-4-6");
        });

        assert_matches!(&events[1], ResponseEvent::OutputTextDelta(text) => {
            assert_eq!(text, "Hello");
        });

        assert_matches!(&events[2], ResponseEvent::OutputTextDelta(text) => {
            assert_eq!(text, " world");
        });

        assert_matches!(&events[3], ResponseEvent::OutputItemDone(
            ResponseItem::Message { role, content, .. }
        ) => {
            assert_eq!(role, "assistant");
            assert_eq!(content.len(), 1);
            assert_matches!(&content[0], ContentItem::OutputText { text } => {
                assert_eq!(text, "Hello world");
            });
        });

        let last = events.last().unwrap();
        assert_matches!(last, ResponseEvent::Completed {
            response_id,
            end_turn,
            ..
        } => {
            assert_eq!(response_id, "msg_001");
            assert_eq!(*end_turn, Some(true));
        });
    }

    #[tokio::test]
    async fn parses_tool_use() {
        let events = collect_messages_events(vec![
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_002",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": { "input_tokens": 100, "output_tokens": 0 }
                }
            }),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "tool_use", "id": "toolu_001", "name": "read_file" }
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "{\"path\": " }
            }),
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "\"/tmp/foo\"}" }
            }),
            json!({
                "type": "content_block_stop",
                "index": 0
            }),
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "tool_use" },
                "usage": { "output_tokens": 20 }
            }),
            json!({
                "type": "message_stop"
            }),
        ])
        .await;

        // Check we got an OutputItemAdded for the tool call
        let added = events
            .iter()
            .find(|e| matches!(e, ResponseEvent::OutputItemAdded(..)));
        assert!(added.is_some(), "expected OutputItemAdded event");

        // Check we got tool call deltas
        let deltas: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, ResponseEvent::ToolCallInputDelta { .. }))
            .collect();
        assert_eq!(deltas.len(), 2);

        // Check the completed tool call
        let done = events.iter().find(|e| {
            matches!(
                e,
                ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { .. })
            )
        });
        assert!(done.is_some(), "expected OutputItemDone with FunctionCall");
        if let Some(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        })) = done
        {
            assert_eq!(name, "read_file");
            assert_eq!(call_id, "toolu_001");
            assert_eq!(arguments, "{\"path\": \"/tmp/foo\"}");
        }
    }

    #[tokio::test]
    async fn ping_before_message_start_is_ignored() {
        let events = collect_messages_events(vec![
            json!({ "type": "ping" }),
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_003",
                    "model": "claude-sonnet-4-6",
                    "role": "assistant",
                    "usage": { "input_tokens": 5, "output_tokens": 0 }
                }
            }),
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "ok" }
            }),
            json!({
                "type": "content_block_stop",
                "index": 0
            }),
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 2 }
            }),
            json!({
                "type": "message_stop"
            }),
        ])
        .await;

        // Should still parse correctly (ping ignored, not errored).
        let completed = events
            .iter()
            .any(|e| matches!(e, ResponseEvent::Completed { .. }));
        assert!(completed, "expected Completed event despite ping before message_start");
    }
}