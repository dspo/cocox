//! Anthropic Messages API prompt caching (`cache_control` breakpoints)。
//!
//! 在请求上放置 **1 个** ephemeral 断点：最后一个 `tool`。该断点缓存其前的
//! system + 全部 tools 稳定前缀，跨 turn 命中读缓存（百炼实测每轮 read≈7800
//! tokens、creation=0）。
//!
//! 为什么只放 tool 一个断点（不放 system / messages）：
//! - **system 保持 `Text` 字符串**：实测百炼 `/apps/anthropic` 在完整请求体下以
//!   `InvalidParameter` 拒绝 system-as-blocks；system 字符串 + tool 断点被接受，
//!   且 tool 断点已隐式覆盖 system 前缀。
//! - **不在 messages 上放断点**：实测百炼只认**最后一个** cache_control 断点。
//!   若在 messages[-1]（当前消息，每轮变）上放断点，百炼每轮重写该前缀、从不读
//!   （creation≈7905/turn、read=0，1.25x 写入成本、净负收益）。只放 tool 断点
//!   （稳定）时，百炼每轮 read≈7808、creation=0。
//! - real Anthropic 原生 API 会 honor 每个断点，messages 断点对长对话前缀缓存
//!   有价值；但 cox 当前主力端是百炼，tool-only 是实测最优的通用默认。

use crate::anthropic::{
    AnthropicCacheControl, AnthropicContentBlockParam, AnthropicMessageContent,
    AnthropicMessageParam, AnthropicMessagesRequest,
};
use serde_json::Value;

/// Anthropic 每个请求最多 4 个 `cache_control` 断点（我们只用 1 个）。
const MAX_CACHE_BREAKPOINTS: usize = 4;

/// short-lived ephemeral 断点：`{type:"ephemeral"}`（5min，无 `ttl`）。
/// 百炼等第三方 anthropic 兼容端不支持 `ttl:"1h"`，统一用 short。
pub fn ephemeral_cache_control() -> AnthropicCacheControl {
    AnthropicCacheControl {
        cache_type: "ephemeral".into(),
        ttl: None,
    }
}

/// 在请求上放置 cache_control 断点：仅最后一个 tool（稳定前缀 system+tools）。
///
/// system 不动（保持字符串；由 tool 断点隐式缓存其前缀）。messages 不动。无 tools
/// 时不动请求（不缓存——cox 通常都带 tools）。
pub fn apply_prompt_caching(request: &mut AnthropicMessagesRequest) {
    let cc = ephemeral_cache_control();

    // 最后一个 tool（Vec<Value>，在对象 Map 里塞 "cache_control" 键）。
    // 该断点缓存其前的 system + 全部 tools 前缀。
    if !request.tools.is_empty()
        && let Some(Value::Object(map)) = request.tools.last_mut()
        && let Ok(cc_val) = serde_json::to_value(&cc)
    {
        map.insert("cache_control".to_string(), cc_val);
    }

    enforce_cache_control_limit(request, MAX_CACHE_BREAKPOINTS);
}

/// 数请求里所有 `Some(cache_control)`（tools + messages blocks；system 不标）。
fn count_breakpoints(request: &AnthropicMessagesRequest) -> usize {
    let mut n = 0usize;
    n += request
        .tools
        .iter()
        .filter(|t| matches!(t, Value::Object(m) if m.contains_key("cache_control")))
        .count();
    for msg in &request.messages {
        if let AnthropicMessageContent::Blocks(blocks) = &msg.content {
            n += count_block_breakpoints(blocks);
        }
    }
    n
}

fn count_block_breakpoints(blocks: &[AnthropicContentBlockParam]) -> usize {
    blocks
        .iter()
        .filter(|b| match b {
            AnthropicContentBlockParam::Text { cache_control, .. }
            | AnthropicContentBlockParam::Image { cache_control, .. } => cache_control.is_some(),
            _ => false,
        })
        .count()
}

/// 超限时剥离断点（omp `enforceCacheControlLimit`）：先 messages 从前到后清，
/// 再清 tools 非末个。正常路径（apply 后 = 1）为 no-op；仅当上游预置了断点时触发。
fn enforce_cache_control_limit(request: &mut AnthropicMessagesRequest, max: usize) {
    while count_breakpoints(request) > max {
        if let Some(i) = first_message_breakpoint_index(&request.messages) {
            clear_one_message_breakpoint(&mut request.messages[i]);
            continue;
        }
        if clear_non_last_tool_breakpoint(&mut request.tools) {
            continue;
        }
        if let Some(i) = request.messages.iter().rposition(|m| {
            matches!(&m.content, AnthropicMessageContent::Blocks(b) if count_block_breakpoints(b) > 0)
        }) {
            clear_one_message_breakpoint(&mut request.messages[i]);
            continue;
        }
        break;
    }
}

fn first_message_breakpoint_index(messages: &[AnthropicMessageParam]) -> Option<usize> {
    messages.iter().position(|m| {
        matches!(&m.content, AnthropicMessageContent::Blocks(b) if count_block_breakpoints(b) > 0)
    })
}

fn clear_one_message_breakpoint(msg: &mut AnthropicMessageParam) {
    if let AnthropicMessageContent::Blocks(blocks) = &mut msg.content
        && let Some(b) = blocks.iter_mut().find(|b| match b {
            AnthropicContentBlockParam::Text { cache_control, .. }
            | AnthropicContentBlockParam::Image { cache_control, .. } => cache_control.is_some(),
            _ => false,
        })
    {
        match b {
            AnthropicContentBlockParam::Text { cache_control, .. }
            | AnthropicContentBlockParam::Image { cache_control, .. } => {
                *cache_control = None;
            }
            _ => {}
        }
    }
}

fn clear_non_last_tool_breakpoint(tools: &mut [Value]) -> bool {
    let last_idx = tools.len().saturating_sub(1);
    for (i, t) in tools.iter_mut().enumerate() {
        if i != last_idx
            && let Value::Object(map) = t
            && map.remove("cache_control").is_some()
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::{
        AnthropicContentBlockParam, AnthropicMessageContent, AnthropicMessageParam,
        AnthropicMessagesRequest, AnthropicSystemPrompt,
    };
    use serde_json::json;

    fn user_text_msg(text: &str) -> AnthropicMessageParam {
        AnthropicMessageParam {
            role: "user".into(),
            content: AnthropicMessageContent::Text(text.into()),
        }
    }

    fn request_with(
        system: Option<AnthropicSystemPrompt>,
        tools: Vec<Value>,
        messages: Vec<AnthropicMessageParam>,
    ) -> AnthropicMessagesRequest {
        AnthropicMessagesRequest {
            model: "m".into(),
            messages,
            max_tokens: 8192,
            system,
            stream: true,
            temperature: None,
            tools,
            tool_choice: None,
            thinking: None,
            metadata: None,
        }
    }

    fn block_has_cc(b: &[AnthropicContentBlockParam]) -> bool {
        b.iter().any(|x| match x {
            AnthropicContentBlockParam::Text { cache_control, .. }
            | AnthropicContentBlockParam::Image { cache_control, .. } => cache_control.is_some(),
            _ => false,
        })
    }

    #[test]
    fn places_breakpoint_on_last_tool_only() {
        // system 保持 Text 不动；messages 不动；只最后一个 tool 带 cc。
        let system = Some(AnthropicSystemPrompt::Text("sys".into()));
        let tools = vec![
            json!({"name":"a","input_schema":{}}),
            json!({"name":"b","input_schema":{}}),
        ];
        let messages = vec![user_text_msg("first"), user_text_msg("second")];
        let mut req = request_with(system, tools, messages);
        apply_prompt_caching(&mut req);

        assert!(matches!(
            req.system.as_ref().unwrap(),
            AnthropicSystemPrompt::Text(_)
        ));
        assert!(matches!(req.tools.last(), Some(Value::Object(m)) if m.contains_key("cache_control")));
        assert!(!matches!(req.tools.first(), Some(Value::Object(m)) if m.contains_key("cache_control")));
        // messages 不动：Text 仍 Text（无 cc）。
        for m in &req.messages {
            assert!(matches!(m.content, AnthropicMessageContent::Text(_)));
        }
        assert_eq!(count_breakpoints(&req), 1);
    }

    #[test]
    fn no_tools_leaves_request_untouched() {
        let messages = vec![user_text_msg("hi")];
        let mut req = request_with(None, vec![], messages);
        apply_prompt_caching(&mut req);
        assert_eq!(count_breakpoints(&req), 0);
    }

    #[test]
    fn enforces_4_cap_when_upstream_pre_seeded() {
        // 预置 5 条 message，每条末块都带 cc（共 5 个断点，超 4 上限）。
        let messages: Vec<AnthropicMessageParam> = (0..5)
            .map(|i| AnthropicMessageParam {
                role: "user".into(),
                content: AnthropicMessageContent::Blocks(vec![AnthropicContentBlockParam::Text {
                    text: format!("m{i}"),
                    cache_control: Some(ephemeral_cache_control()),
                }]),
            })
            .collect();
        let mut req = request_with(None, vec![], messages);
        apply_prompt_caching(&mut req);
        assert!(count_breakpoints(&req) <= MAX_CACHE_BREAKPOINTS);
    }

    #[test]
    fn ephemeral_cc_is_short_no_ttl() {
        let cc = ephemeral_cache_control();
        assert_eq!(cc.cache_type, "ephemeral");
        assert!(cc.ttl.is_none());
        let v = serde_json::to_value(&cc).unwrap();
        assert_eq!(v, json!({"type":"ephemeral"}));
    }

    // 保留 block_has_cc 引用，避免未使用警告（enforce 路径间接用到计数逻辑）。
    #[test]
    fn block_has_cc_helper_smoke() {
        let b = vec![AnthropicContentBlockParam::Text {
            text: "x".into(),
            cache_control: Some(ephemeral_cache_control()),
        }];
        assert!(block_has_cc(&b));
    }
}
