//! Maps Codex's internal [`ResponsesApiRequest`] to a Chat Completions API
//! request body ([`ChatCompletionsRequest`]).
//!
//! The Responses API and Chat Completions API share a common conceptual
//! model (messages, tools, tool choice) but differ in shape. This module is
//! the adapter: it is lossy by design — Responses-only concepts
//! (`store`, `include`, `prompt_cache_key`, `text` formatting, client
//! metadata, reasoning summary/context) are dropped, and only function tools
//! are carried over (Chat Completions has no built-in web search / image
//! generation tools).

use crate::chat_completions::ChatCompletionAssistantToolCall;
use crate::chat_completions::ChatCompletionAssistantToolCallFunction;
use crate::chat_completions::ChatCompletionContent;
use crate::chat_completions::ChatCompletionContentPart;
use crate::chat_completions::ChatCompletionImageUrl;
use crate::chat_completions::ChatCompletionMessageParam;
use crate::chat_completions::ChatCompletionStreamOptions;
use crate::chat_completions::ChatCompletionTool;
use crate::chat_completions::ChatCompletionToolChoice;
use crate::chat_completions::ChatCompletionToolFunction;
use crate::chat_completions::ChatCompletionsRequest;
use crate::common::Reasoning;
use crate::common::ResponsesApiRequest;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ImageDetail;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use serde_json::Value;
use tracing::debug;

/// Converts a Codex internal request into a Chat Completions API request body.
///
/// `stream` is forced to `true` and `stream_options.include_usage` to `true`
/// so the final chunk carries token usage.
pub fn map_to_chat_completions_request(request: &ResponsesApiRequest) -> ChatCompletionsRequest {
    let mut messages: Vec<ChatCompletionMessageParam> = Vec::new();

    if !request.instructions.is_empty() {
        messages.push(ChatCompletionMessageParam::System {
            content: ChatCompletionContent::Text(request.instructions.clone()),
        });
    }

    for item in &request.input {
        if let Some(msg) = map_response_item(item, &mut messages) {
            messages.push(msg);
        }
    }

    let tools = request
        .tools
        .as_ref()
        .map(|tools| reshape_tools(tools))
        .unwrap_or_default();

    ChatCompletionsRequest {
        model: request.model.clone(),
        messages,
        max_tokens: None,
        max_completion_tokens: None,
        temperature: None,
        tools,
        tool_choice: map_tool_choice(&request.tool_choice),
        parallel_tool_calls: Some(request.parallel_tool_calls),
        stream: true,
        stream_options: ChatCompletionStreamOptions { include_usage: true },
        reasoning_effort: request.reasoning.as_ref().and_then(reasoning_effort_string),
        stop: Vec::new(),
    }
}

/// Maps a single [`ResponseItem`] to a Chat Completions message. Returns
/// `None` for items with no Chat Completions equivalent (e.g. reasoning,
/// local shell calls), which are silently dropped.
///
/// `messages` is taken mutably so a [`ResponseItem::FunctionCall`] can be
/// folded into the preceding assistant message's `tool_calls` (Chat
/// Completions attaches tool calls to the assistant turn, unlike the
/// Responses API which emits them as standalone items).
fn map_response_item(
    item: &ResponseItem,
    messages: &mut Vec<ChatCompletionMessageParam>,
) -> Option<ChatCompletionMessageParam> {
    match item {
        ResponseItem::Message {
            role,
            content,
            ..
        } => {
            let content = content_to_chat(content);
            match role.as_str() {
                "system" => Some(ChatCompletionMessageParam::System { content }),
                "developer" => Some(ChatCompletionMessageParam::Developer { content }),
                "assistant" => Some(ChatCompletionMessageParam::Assistant {
                    content: empty_content_to_none(content),
                    tool_calls: Vec::new(),
                }),
                // "user" and any unknown role are treated as user content.
                _ => Some(ChatCompletionMessageParam::User { content }),
            }
        }
        ResponseItem::FunctionCall {
            name,
            arguments,
            call_id,
            ..
        } => {
            let index = next_assistant_tool_call_index(messages);
            let tool_call = ChatCompletionAssistantToolCall {
                index,
                id: call_id.clone(),
                kind: "function".to_string(),
                function: ChatCompletionAssistantToolCallFunction {
                    name: name.clone(),
                    arguments: arguments.clone(),
                },
            };
            append_assistant_tool_call(tool_call, messages);
            None
        }
        ResponseItem::FunctionCallOutput { call_id, output, .. } => {
            let content = output.body.to_text().unwrap_or_default();
            Some(ChatCompletionMessageParam::Tool {
                tool_call_id: call_id.clone(),
                content,
            })
        }
        // Reasoning, local shell calls, MCP/custom tool calls, and other
        // Codex-specific items have no Chat Completions equivalent.
        other => {
            debug!(
                "dropping ResponseItem with no Chat Completions mapping: {}",
                std::any::type_name::<ResponseItem>()
            );
            let _ = other;
            None
        }
    }
}

/// Converts a Codex message content slice into Chat Completions content.
///
/// Plain single-text content becomes a `Text` string; anything with images or
/// multiple parts becomes a `Parts` array.
fn content_to_chat(content: &[ContentItem]) -> ChatCompletionContent {
    if content.is_empty() {
        return ChatCompletionContent::Text(String::new());
    }

    let has_image = content
        .iter()
        .any(|c| matches!(c, ContentItem::InputImage { .. }));

    if !has_image && content.len() == 1 {
        return ChatCompletionContent::Text(text_of(&content[0]).unwrap_or_default());
    }

    let mut parts = Vec::with_capacity(content.len());
    for item in content {
        match item {
            ContentItem::InputText { text }
            | ContentItem::OutputText { text } => parts.push(ChatCompletionContentPart::Text {
                text: text.clone(),
            }),
            ContentItem::InputImage { image_url, detail } => {
                parts.push(ChatCompletionContentPart::ImageUrl {
                    image_url: ChatCompletionImageUrl {
                        url: image_url.clone(),
                        detail: detail.as_ref().map(image_detail_str),
                    },
                });
            }
        }
    }
    ChatCompletionContent::Parts(parts)
}

fn text_of(item: &ContentItem) -> Option<String> {
    match item {
        ContentItem::InputText { text } | ContentItem::OutputText { text } => Some(text.clone()),
        ContentItem::InputImage { .. } => None,
    }
}

/// Maps Codex's `ImageDetail` to the Chat Completions `detail` string.
fn image_detail_str(detail: &ImageDetail) -> String {
    match detail {
        ImageDetail::Auto => "auto",
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
        ImageDetail::Original => "high",
    }
    .to_string()
}

/// Returns `None` for empty text content so assistant turns with only tool
/// calls serialize `content` as absent (Chat Completions allows null content
/// in that case).
fn empty_content_to_none(content: ChatCompletionContent) -> Option<ChatCompletionContent> {
    match &content {
        ChatCompletionContent::Text(t) if t.is_empty() => None,
        _ => Some(content),
    }
}

/// Appends a tool call to the last assistant message, creating one if needed.
fn append_assistant_tool_call(
    tool_call: ChatCompletionAssistantToolCall,
    messages: &mut Vec<ChatCompletionMessageParam>,
) {
    match messages.last_mut() {
        Some(ChatCompletionMessageParam::Assistant { tool_calls, .. }) => {
            tool_calls.push(tool_call);
        }
        _ => {
            messages.push(ChatCompletionMessageParam::Assistant {
                content: None,
                tool_calls: vec![tool_call],
            });
        }
    }
}

/// Computes the `index` for the next tool call on the last assistant message
/// (Chat Completions aligns streaming deltas by index).
fn next_assistant_tool_call_index(messages: &[ChatCompletionMessageParam]) -> i64 {
    match messages.last() {
        Some(ChatCompletionMessageParam::Assistant { tool_calls, .. }) => {
            tool_calls.last().map(|t| t.index + 1).unwrap_or(0)
        }
        _ => 0,
    }
}

/// Reshapes Responses-API tool JSON (`{"type":"function", name, ...}`) into
/// Chat Completions function-calling shape
/// (`{"type":"function","function":{name, description, parameters, strict}}`).
/// Non-function tools are dropped.
fn reshape_tools(tools: &[Value]) -> Vec<ChatCompletionTool> {
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        let kind = tool.get("type").and_then(Value::as_str).unwrap_or("");
        if kind != "function" {
            debug!("dropping non-function tool from Chat Completions request");
            continue;
        }
        let name = match tool.get("name").and_then(Value::as_str) {
            Some(name) => name.to_string(),
            None => {
                debug!("dropping function tool without a name");
                continue;
            }
        };
        let parameters = tool.get("parameters").cloned().unwrap_or(Value::Object(
            serde_json::Map::new(),
        ));
        out.push(ChatCompletionTool {
            kind: "function".to_string(),
            function: ChatCompletionToolFunction {
                name,
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                parameters,
                strict: tool.get("strict").and_then(Value::as_bool),
            },
        });
    }
    out
}

/// Maps the Responses `tool_choice` string to a Chat Completions tool choice.
fn map_tool_choice(tool_choice: &str) -> Option<ChatCompletionToolChoice> {
    match tool_choice {
        "" => None,
        "auto" | "none" | "required" => Some(ChatCompletionToolChoice::Mode(
            tool_choice.to_string(),
        )),
        // Responses may emit a function name here in some flows; Chat
        // Completions would need a named choice, but we don't have enough
        // structure to build one from a bare string, so fall back to auto.
        _ => Some(ChatCompletionToolChoice::Mode("auto".to_string())),
    }
}

fn reasoning_effort_string(reasoning: &Reasoning) -> Option<String> {
    reasoning.effort.as_ref().map(map_reasoning_effort)
}

fn map_reasoning_effort(effort: &ReasoningEffort) -> String {
    match effort {
        ReasoningEffort::None
        | ReasoningEffort::Minimal => "minimal".to_string(),
        ReasoningEffort::Low => "low".to_string(),
        ReasoningEffort::Medium => "medium".to_string(),
        ReasoningEffort::High => "high".to_string(),
        ReasoningEffort::XHigh => "xhigh".to_string(),
        // Chat Completions has no "ultra"; clamp to the highest known value.
        ReasoningEffort::Ultra => "high".to_string(),
        ReasoningEffort::Custom(value) => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Reasoning;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::ResponseItem;
    use codex_protocol::openai_models::ReasoningEffort;
    use serde_json::json;

    fn user(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".into(),
            content: vec![ContentItem::InputText {
                text: text.into(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn assistant(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "assistant".into(),
            content: vec![ContentItem::OutputText {
                text: text.into(),
            }],
            phase: None,
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn function_call(call_id: &str, name: &str, args: &str) -> ResponseItem {
        ResponseItem::FunctionCall {
            id: None,
            name: name.into(),
            namespace: None,
            arguments: args.into(),
            call_id: call_id.into(),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn tool_output(call_id: &str, output: &str) -> ResponseItem {
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: call_id.into(),
            output: codex_protocol::models::FunctionCallOutputPayload::from_text(output.into()),
            internal_chat_message_metadata_passthrough: None,
        }
    }

    fn basic_request(input: Vec<ResponseItem>) -> ResponsesApiRequest {
        ResponsesApiRequest {
            model: "gpt-4o".into(),
            instructions: String::new(),
            input,
            tools: None,
            tool_choice: "auto".into(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        }
    }

    #[test]
    fn maps_simple_conversation() {
        let req = basic_request(vec![user("hi"), assistant("hello")]);
        let out = map_to_chat_completions_request(&req);
        assert_eq!(out.model, "gpt-4o");
        assert_eq!(out.stream, true);
        assert_eq!(out.stream_options.include_usage, true);
        assert_eq!(out.parallel_tool_calls, Some(true));
        assert!(matches!(
            &out.messages[0],
            ChatCompletionMessageParam::User { .. }
        ));
        assert!(matches!(
            &out.messages[1],
            ChatCompletionMessageParam::Assistant { .. }
        ));
    }

    #[test]
    fn injects_instructions_as_system_message() {
        let mut req = basic_request(vec![user("hi")]);
        req.instructions = "be brief".into();
        let out = map_to_chat_completions_request(&req);
        assert!(matches!(
            &out.messages[0],
            ChatCompletionMessageParam::System { .. }
        ));
        assert_eq!(out.messages.len(), 2);
    }

    #[test]
    fn folds_function_calls_into_assistant_tool_calls() {
        let req = basic_request(vec![
            function_call("call_1", "search", "{\"q\":\"rust\"}"),
            tool_output("call_1", "results"),
        ]);
        let out = map_to_chat_completions_request(&req);
        // Assistant message with one tool call, followed by a tool result.
        assert_eq!(out.messages.len(), 2);
        let assistant = match &out.messages[0] {
            ChatCompletionMessageParam::Assistant { tool_calls, .. } => tool_calls,
            _ => panic!("expected assistant with tool_calls"),
        };
        assert_eq!(assistant.len(), 1);
        assert_eq!(assistant[0].index, 0);
        assert_eq!(assistant[0].id, "call_1");
        assert_eq!(assistant[0].function.name, "search");
        assert_eq!(assistant[0].function.arguments, "{\"q\":\"rust\"}");

        match &out.messages[1] {
            ChatCompletionMessageParam::Tool {
                tool_call_id,
                content,
            } => {
                assert_eq!(tool_call_id, "call_1");
                assert_eq!(content, "results");
            }
            _ => panic!("expected tool message"),
        }
    }

    #[test]
    fn assigns_sequential_tool_call_indices() {
        let req = basic_request(vec![
            function_call("call_1", "a", "{}"),
            function_call("call_2", "b", "{}"),
        ]);
        let out = map_to_chat_completions_request(&req);
        let tool_calls = match &out.messages[0] {
            ChatCompletionMessageParam::Assistant { tool_calls, .. } => tool_calls,
            _ => panic!("expected assistant"),
        };
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].index, 0);
        assert_eq!(tool_calls[1].index, 1);
    }

    #[test]
    fn reshapes_function_tools() {
        let mut req = basic_request(vec![user("hi")]);
        req.tools = Some(vec![
            json!({
                "type": "function",
                "name": "search",
                "description": "search the web",
                "parameters": {"type": "object"},
                "strict": true,
            }),
            json!({"type": "web_search"}),
        ]);
        let out = map_to_chat_completions_request(&req);
        assert_eq!(out.tools.len(), 1);
        assert_eq!(out.tools[0].kind, "function");
        assert_eq!(out.tools[0].function.name, "search");
        assert_eq!(out.tools[0].function.strict, Some(true));
        assert_eq!(out.tools[0].function.parameters, json!({"type":"object"}));
    }

    #[test]
    fn maps_reasoning_effort() {
        let mut req = basic_request(vec![user("hi")]);
        req.reasoning = Some(Reasoning {
            effort: Some(ReasoningEffort::High),
            summary: None,
            context: None,
        });
        let out = map_to_chat_completions_request(&req);
        assert_eq!(out.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn drops_reasoning_items() {
        let reasoning = ResponseItem::Reasoning {
            id: None,
            summary: Vec::new(),
            content: None,
            encrypted_content: None,
            internal_chat_message_metadata_passthrough: None,
        };
        let req = basic_request(vec![reasoning, user("hi")]);
        let out = map_to_chat_completions_request(&req);
        // Only the user message survives.
        assert_eq!(out.messages.len(), 1);
        assert!(matches!(
            &out.messages[0],
            ChatCompletionMessageParam::User { .. }
        ));
    }
}
