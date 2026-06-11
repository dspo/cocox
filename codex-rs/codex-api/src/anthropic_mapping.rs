//! Conversion between internal `ResponseItem`/`ContentItem` and Anthropic
//! Messages API content blocks.
//!
//! This module converts a Codex conversation history (expressed as
//! `Vec<ResponseItem>`) into the `messages` array expected by
//! `AnthropicMessagesRequest`.

use crate::anthropic::AnthropicContentBlockParam;
use crate::anthropic::AnthropicImageSource;
use crate::anthropic::AnthropicMessageContent;
use crate::anthropic::AnthropicMessageParam;
use crate::anthropic::AnthropicToolResultContent;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::ResponseItem;
use serde_json::Value;

/// Converts a list of Codex `ResponseItem`s into the `messages` array for the
/// Anthropic Messages API, handling role adjacency constraints.
///
/// ## Rules enforced
/// - `tool_result` must be wrapped in a `role: "user"` message.
/// - `tool_use` blocks must appear in a `role: "assistant"` message.
/// - Adjacent messages with the same role are merged where possible.
pub fn response_items_to_anthropic_messages(
    items: &[ResponseItem],
) -> Vec<AnthropicMessageParam> {
    let mut messages: Vec<AnthropicMessageParam> = Vec::new();

    for item in items {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let blocks = content_items_to_blocks(content);
                if blocks.is_empty() {
                    continue;
                }
                let new_msg = AnthropicMessageParam {
                    role: role.clone(),
                    content: AnthropicMessageContent::Blocks(blocks),
                };
                push_or_merge(&mut messages, new_msg);
            }
            ResponseItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                let tool_use = AnthropicContentBlockParam::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input: parse_arguments(&arguments),
                };
                let new_msg = AnthropicMessageParam {
                    role: "assistant".to_string(),
                    content: AnthropicMessageContent::Blocks(vec![tool_use]),
                };
                push_or_merge(&mut messages, new_msg);
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                let result_content = function_output_to_tool_result_content(output);
                let tool_result = AnthropicContentBlockParam::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: result_content,
                    is_error: None,
                };
                let new_msg = AnthropicMessageParam {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::Blocks(vec![tool_result]),
                };
                // Tool results always create a separate user message.
                messages.push(new_msg);
            }
            ResponseItem::CustomToolCall {
                call_id,
                name,
                input,
                ..
            } => {
                let tool_use = AnthropicContentBlockParam::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input: serde_json::from_str(&input).unwrap_or(Value::Null),
                };
                let new_msg = AnthropicMessageParam {
                    role: "assistant".to_string(),
                    content: AnthropicMessageContent::Blocks(vec![tool_use]),
                };
                push_or_merge(&mut messages, new_msg);
            }
            ResponseItem::CustomToolCallOutput {
                call_id, output, ..
            } => {
                let result_content = function_output_to_tool_result_content(output);
                let tool_result = AnthropicContentBlockParam::ToolResult {
                    tool_use_id: call_id.clone(),
                    content: result_content,
                    is_error: None,
                };
                let new_msg = AnthropicMessageParam {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::Blocks(vec![tool_result]),
                };
                messages.push(new_msg);
            }
            // Compaction, reasoning, image generation etc. are not representable
            // in Anthropic Messages — skip them.
            _ => {}
        }
    }

    messages
}

/// Converts Codex `ContentItem`s into Anthropic content blocks.
fn content_items_to_blocks(content: &[ContentItem]) -> Vec<AnthropicContentBlockParam> {
    content
        .iter()
        .filter_map(|item| match item {
            ContentItem::InputText { text } => Some(AnthropicContentBlockParam::Text {
                text: text.clone(),
                cache_control: None,
            }),
            ContentItem::OutputText { text } => Some(AnthropicContentBlockParam::Text {
                text: text.clone(),
                cache_control: None,
            }),
            ContentItem::InputImage { image_url, detail: _ } => {
                // Anthropic supports URL-based images.
                Some(AnthropicContentBlockParam::Image {
                    source: AnthropicImageSource::Url {
                        url: image_url.clone(),
                    },
                    cache_control: None,
                })
            }
        })
        .collect()
}

/// Converts a function call output into Anthropic tool result content.
fn function_output_to_tool_result_content(
    output: &codex_protocol::models::FunctionCallOutputPayload,
) -> AnthropicToolResultContent {
    match &output.body {
        FunctionCallOutputBody::Text(text) => {
            AnthropicToolResultContent::Text(text.clone())
        }
        FunctionCallOutputBody::ContentItems(items) => {
            AnthropicToolResultContent::Text(format_content_items(items))
        }
    }
}

fn format_content_items(
    items: &[codex_protocol::models::FunctionCallOutputContentItem],
) -> String {
    items
        .iter()
        .filter_map(|item| match item {
            codex_protocol::models::FunctionCallOutputContentItem::InputText { text } => {
                Some(text.clone())
            }
            codex_protocol::models::FunctionCallOutputContentItem::InputImage {
                image_url, ..
            } => Some(format!("[image: {image_url}]")),
            codex_protocol::models::FunctionCallOutputContentItem::EncryptedContent { .. } => {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Merges the new message into the previous one if their roles match; otherwise
/// pushes it as a new entry.
fn push_or_merge(messages: &mut Vec<AnthropicMessageParam>, new_msg: AnthropicMessageParam) {
    if let Some(last) = messages.last_mut()
        && last.role == new_msg.role
    {
        // Merge content blocks.
        if let (
            AnthropicMessageContent::Blocks(existing_blocks),
            AnthropicMessageContent::Blocks(mut new_blocks),
        ) = (&mut last.content, new_msg.content)
        {
            existing_blocks.append(&mut new_blocks);
        }
    } else {
        messages.push(new_msg);
    }
}

/// Tries to parse arguments JSON string into a Value; returns Null on failure.
fn parse_arguments(arguments: &str) -> Value {
    if arguments.is_empty() {
        return Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(arguments).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::ResponseItem;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    #[test]
    fn converts_user_text_message() {
        let items = vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "hello".to_string(),
            }],
            phase: None,
        }];

        let messages = response_items_to_anthropic_messages(&items);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        match &messages[0].content {
            AnthropicMessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    AnthropicContentBlockParam::Text { text, .. } => {
                        assert_eq!(text, "hello");
                    }
                    _ => panic!("expected Text block"),
                }
            }
            _ => panic!("expected Blocks content"),
        }
    }

    #[test]
    fn merges_adjacent_same_role_messages() {
        let items = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "first".to_string(),
                }],
                phase: None,
            },
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "second".to_string(),
                }],
                phase: None,
            },
        ];

        let messages = response_items_to_anthropic_messages(&items);
        assert_eq!(messages.len(), 1, "adjacent user messages should merge");
        assert_eq!(messages[0].role, "user");
        match &messages[0].content {
            AnthropicMessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
            }
            _ => panic!("expected Blocks content"),
        }
    }

    #[test]
    fn tool_use_and_result_separate_messages() {
        let items = vec![
            ResponseItem::FunctionCall {
                id: None,
                name: "read_file".to_string(),
                namespace: None,
                arguments: r#"{"path":"/tmp/x"}"#.to_string(),
                call_id: "toolu_01".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "toolu_01".to_string(),
                output: FunctionCallOutputPayload::from_text("file contents".to_string()),
            },
        ];

        let messages = response_items_to_anthropic_messages(&items);
        assert_eq!(messages.len(), 2);

        // First message is the tool use (assistant)
        assert_eq!(messages[0].role, "assistant");
        match &messages[0].content {
            AnthropicMessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    AnthropicContentBlockParam::ToolUse { id, name, input } => {
                        assert_eq!(id, "toolu_01");
                        assert_eq!(name, "read_file");
                        assert_eq!(input, &json!({"path": "/tmp/x"}));
                    }
                    _ => panic!("expected ToolUse block"),
                }
            }
            _ => panic!("expected Blocks content"),
        }

        // Second message is the tool result (user)
        assert_eq!(messages[1].role, "user");
        match &messages[1].content {
            AnthropicMessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    AnthropicContentBlockParam::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        assert_eq!(tool_use_id, "toolu_01");
                        assert_matches::assert_matches!(
                            content,
                            AnthropicToolResultContent::Text(t) if t == "file contents"
                        );
                    }
                    _ => panic!("expected ToolResult block"),
                }
            }
            _ => panic!("expected Blocks content"),
        }
    }
}