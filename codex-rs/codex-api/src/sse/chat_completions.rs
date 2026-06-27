//! SSE parser for the OpenAI Chat Completions streaming endpoint.
//!
//! Converts the Chat Completions SSE chunk stream into Codex's internal
//! [`ResponseEvent`] stream. Unlike the Responses API, Chat Completions
//! streams *deltas*: text and tool-call arguments arrive incrementally and
//! must be aggregated client-side into complete [`ResponseItem`]s.
//!
//! The synthesized event sequence mirrors what a Responses-API stream would
//! produce so the upstream core logic is wire-protocol agnostic:
//!
//! 1. `ResponseEvent::Created` — emitted once on the first meaningful chunk.
//! 2. `ResponseEvent::OutputTextDelta` — per `delta.content` fragment.
//! 3. `ResponseEvent::OutputItemAdded(FunctionCall)` — when a tool call's
//!    first argument fragment arrives (id + name are known by then).
//! 4. `ResponseEvent::ToolCallInputDelta` — per `function.arguments` fragment.
//! 5. `ResponseEvent::OutputItemDone(Message | FunctionCall)` — flushed when
//!    `finish_reason` appears (or on `[DONE]` as a safety net).
//! 6. `ResponseEvent::Completed` — emitted on the terminating `[DONE]` line,
//!    carrying token usage from the final chunk.

use crate::chat_completions::ChatCompletionChunk;
use crate::chat_completions::ChatCompletionUsage;
use crate::chat_completions::STREAM_DONE_SENTINEL;
use crate::chat_completions::finish_reason_ends_turn;
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
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

/// Per-tool-call accumulator. Chat Completions may interleave fragments for
/// several tool calls (keyed by `index`), so we track them in a map rather
/// than a single active slot.
#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: Option<String>,
    name: Option<String>,
    /// Incrementally appended JSON argument string.
    arguments: String,
    /// Whether `OutputItemAdded` has already been emitted for this call.
    announced: bool,
}

/// State tracked across a single Chat Completions streaming turn.
#[derive(Debug, Default)]
struct ChatCompletionsStreamState {
    response_id: Option<String>,
    /// Most recently reported server model; used to dedupe `ServerModel` events.
    server_model: Option<String>,
    text_buffer: String,
    tool_calls: BTreeMap<i64, ToolCallAccumulator>,
    usage: Option<ChatCompletionUsage>,
    finish_reason: Option<String>,
    created_emitted: bool,
    /// Whether aggregated items have been flushed as `OutputItemDone`.
    flushed: bool,
}

pub fn spawn_chat_completion_stream(
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
        process_chat_completion_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

/// Drives the SSE event loop for the Chat Completions API, translating chunks
/// into `ResponseEvent`s.
async fn process_chat_completion_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = ChatCompletionsStreamState::default();

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }

        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Chat Completions SSE error: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                // Stream closed without a `[DONE]` sentinel. If we already
                // saw a `finish_reason`, synthesize completion; otherwise this
                // is an abnormal close.
                if state.flushed {
                    let _ = tx_event.send(Ok(completed_event(&state))).await;
                } else {
                    let _ = tx_event
                        .send(Err(ApiError::Stream(
                            "chat completions stream closed before [DONE]".into(),
                        )))
                        .await;
                }
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "idle timeout waiting for Chat Completions SSE".into(),
                    )))
                    .await;
                return;
            }
        };

        trace!("Chat Completions SSE event: {}", &sse.data);

        if sse.data.trim() == STREAM_DONE_SENTINEL {
            let mut events = Vec::new();
            flush(&mut state, &mut events);
            events.push(completed_event(&state));
            for event in events {
                let is_completed = matches!(event, ResponseEvent::Completed { .. });
                if tx_event.send(Ok(event)).await.is_err() {
                    return;
                }
                if is_completed {
                    return;
                }
            }
            return;
        }

        let chunk: ChatCompletionChunk = match serde_json::from_str(&sse.data) {
            Ok(chunk) => chunk,
            Err(e) => {
                debug!(
                    "Failed to parse Chat Completions SSE chunk: {e}, data: {}",
                    &sse.data
                );
                continue;
            }
        };

        let events = process_chat_completion_chunk(chunk, &mut state);
        for event in events {
            let is_completed = matches!(event, ResponseEvent::Completed { .. });
            if tx_event.send(Ok(event)).await.is_err() {
                return;
            }
            if is_completed {
                return;
            }
        }
    }
}

/// Processes a single Chat Completions chunk, updating `state` and returning
/// any `ResponseEvent`s to emit.
fn process_chat_completion_chunk(
    chunk: ChatCompletionChunk,
    state: &mut ChatCompletionsStreamState,
) -> Vec<ResponseEvent> {
    let mut events = Vec::new();

    if state.response_id.is_none() {
        state.response_id = chunk.id;
    }
    if let Some(usage) = chunk.usage {
        state.usage = Some(usage);
    }
    if let Some(model) = chunk.model
        && state.server_model.as_deref() != Some(model.as_str())
    {
        state.server_model = Some(model.clone());
        events.push(ResponseEvent::ServerModel(model));
    }

    for choice in chunk.choices {
        // Emit `Created` once on the first chunk carrying actual content.
        if !state.created_emitted {
            let delta = &choice.delta;
            if delta.role.is_some()
                || delta.content.as_deref().is_some_and(|c| !c.is_empty())
                || !delta.tool_calls.is_empty()
            {
                events.push(ResponseEvent::Created);
                state.created_emitted = true;
            }
        }

        if let Some(content) = choice.delta.content
            && !content.is_empty()
        {
            events.push(ResponseEvent::OutputTextDelta(content.clone()));
            state.text_buffer.push_str(&content);
        }

        for tc in choice.delta.tool_calls {
            let acc = state.tool_calls.entry(tc.index).or_default();
            if let Some(id) = tc.id {
                acc.id = Some(id);
            }
            if let Some(function) = tc.function {
                if let Some(name) = function.name {
                    acc.name = Some(name);
                }
                if let Some(arguments) = function.arguments {
                    if !arguments.is_empty() {
                        if !acc.announced {
                            acc.announced = true;
                            events.push(ResponseEvent::OutputItemAdded(
                                ResponseItem::FunctionCall {
                                    id: None,
                                    name: acc.name.clone().unwrap_or_default(),
                                    namespace: None,
                                    arguments: String::new(),
                                    call_id: acc.id.clone().unwrap_or_default(),
                                    internal_chat_message_metadata_passthrough: None,
                                },
                            ));
                        }
                        events.push(ResponseEvent::ToolCallInputDelta {
                            item_id: acc
                                .id
                                .clone()
                                .unwrap_or_else(|| tc.index.to_string()),
                            call_id: acc.id.clone(),
                            delta: arguments.clone(),
                        });
                        acc.arguments.push_str(&arguments);
                    }
                }
            }
        }

        if let Some(finish_reason) = choice.finish_reason
            && state.finish_reason.is_none()
        {
            state.finish_reason = Some(finish_reason);
        }
    }

    // Once the model signals completion, flush the aggregated items so that
    // `OutputItemDone` arrives before `Completed`.
    if state.finish_reason.is_some() && !state.flushed {
        flush(state, &mut events);
    }

    events
}

/// Flushes accumulated text and tool calls as `OutputItemDone` events.
fn flush(state: &mut ChatCompletionsStreamState, events: &mut Vec<ResponseEvent>) {
    if state.flushed {
        return;
    }
    state.flushed = true;

    if !state.text_buffer.is_empty() {
        let text = std::mem::take(&mut state.text_buffer);
        events.push(ResponseEvent::OutputItemDone(ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText { text }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }));
    }

    for (index, acc) in std::mem::take(&mut state.tool_calls) {
        events.push(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
            id: None,
            name: acc.name.unwrap_or_default(),
            namespace: None,
            arguments: acc.arguments,
            call_id: acc.id.unwrap_or_else(|| index.to_string()),
            internal_chat_message_metadata_passthrough: None,
        }));
    }
}

/// Builds the terminal `Completed` event from the accumulated state.
fn completed_event(state: &ChatCompletionsStreamState) -> ResponseEvent {
    ResponseEvent::Completed {
        response_id: state.response_id.clone().unwrap_or_default(),
        token_usage: state.usage.as_ref().map(TokenUsage::from),
        end_turn: state
            .finish_reason
            .as_deref()
            .and_then(finish_reason_ends_turn),
    }
}

impl From<&ChatCompletionUsage> for TokenUsage {
    fn from(usage: &ChatCompletionUsage) -> Self {
        TokenUsage {
            input_tokens: usage.prompt_tokens,
            cached_input_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens)
                .unwrap_or(0),
            output_tokens: usage.completion_tokens,
            reasoning_output_tokens: usage
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens)
                .unwrap_or(0),
            total_tokens: usage.total_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat_completions::ChatCompletionChoice;
    use crate::chat_completions::ChatCompletionDelta;
    use crate::chat_completions::ChatCompletionToolCallDelta;
    use crate::chat_completions::ChatCompletionToolCallFunctionDelta;
    use serde_json::json;

    fn chunk(id: &str, choices: Vec<ChatCompletionChoice>, usage: Option<ChatCompletionUsage>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: Some(id.to_string()),
            model: Some("gpt-4o".into()),
            choices,
            usage,
        }
    }

    fn text_delta(content: &str) -> ChatCompletionChoice {
        ChatCompletionChoice {
            delta: ChatCompletionDelta {
                role: None,
                content: Some(content.into()),
                tool_calls: Vec::new(),
            },
            finish_reason: None,
        }
    }

    fn tool_call_delta(index: i64, id: Option<&str>, name: Option<&str>, args: Option<&str>) -> ChatCompletionChoice {
        ChatCompletionChoice {
            delta: ChatCompletionDelta {
                role: None,
                content: None,
                tool_calls: vec![ChatCompletionToolCallDelta {
                    index,
                    id: id.map(str::to_string),
                    function: Some(ChatCompletionToolCallFunctionDelta {
                        name: name.map(str::to_string),
                        arguments: args.map(str::to_string),
                    }),
                }],
            },
            finish_reason: None,
        }
    }

    fn finish(reason: &str) -> ChatCompletionChoice {
        ChatCompletionChoice {
            delta: ChatCompletionDelta::default(),
            finish_reason: Some(reason.into()),
        }
    }

    #[test]
    fn streams_text_then_completes() {
        let mut state = ChatCompletionsStreamState::default();

        // First chunk: role + first text fragment.
        let mut first = chunk("chatcmpl-1", vec![text_delta("Hello")], None);
        first.choices[0].delta.role = Some("assistant".into());
        let events = process_chat_completion_chunk(first, &mut state);
        assert!(events.iter().any(|e| matches!(e, ResponseEvent::Created)));
        assert!(events.iter().any(|e| matches!(
            e,
            ResponseEvent::OutputTextDelta(t) if t == "Hello"
        )));

        // Second chunk: more text.
        let events = process_chat_completion_chunk(chunk("chatcmpl-1", vec![text_delta(" world")], None), &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            ResponseEvent::OutputTextDelta(t) if t == " world"
        )));

        // Final chunk: finish_reason.
        let events = process_chat_completion_chunk(chunk("chatcmpl-1", vec![finish("stop")], None), &mut state);
        let done = events.iter().find_map(|e| match e {
            ResponseEvent::OutputItemDone(ResponseItem::Message { content, .. }) => {
                Some(content.clone())
            }
            _ => None,
        });
        let content = done.expect("expected OutputItemDone Message");
        match &content[0] {
            ContentItem::OutputText { text } => assert_eq!(text, "Hello world"),
            other => panic!("expected OutputText, got {other:?}"),
        }

        // Completed event.
        let completed = completed_event(&state);
        match completed {
            ResponseEvent::Completed {
                response_id,
                token_usage,
                end_turn,
            } => {
                assert_eq!(response_id, "chatcmpl-1");
                assert!(token_usage.is_none());
                assert_eq!(end_turn, Some(true));
            }
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn streams_tool_call_then_completes() {
        let mut state = ChatCompletionsStreamState::default();

        // First tool-call delta: id + name + first argument fragment.
        let events = process_chat_completion_chunk(
            chunk("chatcmpl-2", vec![tool_call_delta(0, Some("call_1"), Some("search"), Some("{\"q\":"))], None),
            &mut state,
        );
        assert!(events.iter().any(|e| matches!(
            e,
            ResponseEvent::OutputItemAdded(ResponseItem::FunctionCall { name, .. }) if name == "search"
        )));
        let delta_count = events
            .iter()
            .filter(|e| matches!(e, ResponseEvent::ToolCallInputDelta { .. }))
            .count();
        assert_eq!(delta_count, 1);

        // Second fragment: only arguments.
        let events = process_chat_completion_chunk(
            chunk("chatcmpl-2", vec![tool_call_delta(0, None, None, Some("\"rust\"}"))], None),
            &mut state,
        );
        assert!(events
            .iter()
            .filter(|e| matches!(e, ResponseEvent::ToolCallInputDelta { .. }))
            .any(|e| matches!(e, ResponseEvent::ToolCallInputDelta { delta, .. } if delta == "\"rust\"}")));

        // Finish.
        let events = process_chat_completion_chunk(chunk("chatcmpl-2", vec![finish("tool_calls")], None), &mut state);
        let done = events.iter().find_map(|e| match e {
            ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { name, arguments, call_id, .. }) => {
                Some((name.clone(), arguments.clone(), call_id.clone()))
            }
            _ => None,
        });
        let (name, arguments, call_id) = done.expect("expected OutputItemDone FunctionCall");
        assert_eq!(name, "search");
        assert_eq!(arguments, "{\"q\":\"rust\"}");
        assert_eq!(call_id, "call_1");

        match completed_event(&state) {
            ResponseEvent::Completed { end_turn, .. } => assert_eq!(end_turn, Some(false)),
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn captures_usage_from_final_chunk() {
        let mut state = ChatCompletionsStreamState::default();
        process_chat_completion_chunk(chunk("c", vec![text_delta("hi")], None), &mut state);
        process_chat_completion_chunk(chunk("c", vec![finish("stop")], None), &mut state);

        let usage = ChatCompletionUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: Some(crate::chat_completions::ChatCompletionPromptTokensDetails {
                cached_tokens: 4,
            }),
            completion_tokens_details: None,
        };
        // Usage typically arrives on a separate empty-choices chunk.
        process_chat_completion_chunk(chunk("c", Vec::new(), Some(usage)), &mut state);

        match completed_event(&state) {
            ResponseEvent::Completed {
                token_usage: Some(tu),
                ..
            } => {
                assert_eq!(tu.input_tokens, 10);
                assert_eq!(tu.cached_input_tokens, 4);
                assert_eq!(tu.output_tokens, 5);
                assert_eq!(tu.total_tokens, 15);
            }
            other => panic!("expected Completed with usage, got {other:?}"),
        }
    }

    #[test]
    fn flushes_on_done_without_finish_reason() {
        // Some servers omit finish_reason but still send [DONE]; ensure we flush.
        let mut state = ChatCompletionsStreamState::default();
        process_chat_completion_chunk(chunk("c", vec![text_delta("hi")], None), &mut state);
        assert!(!state.flushed);

        let mut events = Vec::new();
        flush(&mut state, &mut events);
        assert!(state.flushed);
        assert!(events.iter().any(|e| matches!(
            e,
            ResponseEvent::OutputItemDone(ResponseItem::Message { .. })
        )));
    }

    #[test]
    fn parses_real_sse_payload_shapes() {
        let payload = json!({
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": { "content": "Hi" },
                "finish_reason": null
            }]
        })
        .to_string();
        let chunk: ChatCompletionChunk = serde_json::from_str(&payload).unwrap();
        let mut state = ChatCompletionsStreamState::default();
        let events = process_chat_completion_chunk(chunk, &mut state);
        assert!(events.iter().any(|e| matches!(
            e,
            ResponseEvent::OutputTextDelta(t) if t == "Hi"
        )));
    }
}
