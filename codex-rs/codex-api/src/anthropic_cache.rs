//! Anthropic Messages API prompt caching (`cache_control` breakpoints)。
//!
//! 策略由 provider 能力决定（`PromptCachingPolicy`），调用方传入：
//! - `Full`：system 末块 + 最后 tool + messages[-2] + messages[-1]，4 断点。
//!   适用于 honor 所有断点的 provider（真 Anthropic 等）。
//! - `LastBreakpointOnly`：仅最后 tool（稳定前缀 system+tools）。
//!   适用于只认最后一个断点的 provider（百炼 /apps/anthropic 等）——
//!   实测百炼在 messages[-1]（每轮变）上放断点会每轮重写、从不读（净负），
//!   只放 tool 断点时每轮 read≈8000、creation=0。
//! - `None`：不缓存。
//!
//! 默认策略由 `resolve_prompt_caching_policy` 按 `ModelProviderInfo` 决定：
//! config 的 `prompt_caching` 显式优先；否则 `api.anthropic.com` → Full，
//! 其余第三方 → LastBreakpointOnly（保守，第三方 anthropic 兼容端常有断点限制）。
//! 这与 omp `supportsLongCacheRetention` 的"官方端给满配、第三方保守"哲学一致。

use crate::anthropic::{
    AnthropicCacheControl, AnthropicContentBlockParam, AnthropicMessageContent,
    AnthropicMessageParam, AnthropicMessagesRequest, AnthropicSystemPrompt,
    AnthropicTextBlockParam,
};
use serde_json::Value;

/// Anthropic 每个请求最多 4 个 `cache_control` 断点。
const MAX_CACHE_BREAKPOINTS: usize = 4;

/// short-lived ephemeral 断点：`{type:"ephemeral"}`（5min，无 `ttl`）。
/// 第三方端通常不支持 `ttl:"1h"`，统一 short；真 Anthropic 的 long TTL 留待后续。
pub fn ephemeral_cache_control() -> AnthropicCacheControl {
    AnthropicCacheControl {
        cache_type: "ephemeral".into(),
        ttl: None,
    }
}

/// Prompt caching 策略，由 provider 能力决定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptCachingPolicy {
    /// 不缓存。
    None,
    /// 完整 4 断点：system 末块 + 最后 tool + messages[-2] + messages[-1]。
    Full,
    /// 仅最后 tool 断点（稳定前缀 system+tools）。
    LastBreakpointOnly,
}

impl PromptCachingPolicy {
    /// 从 config 字符串解析（"none"/"full"/"last_breakpoint"）。
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "none" => Some(Self::None),
            "full" => Some(Self::Full),
            "last_breakpoint" | "last-breakpoint" => Some(Self::LastBreakpointOnly),
            _ => None,
        }
    }
}

/// 按 provider 能力解析策略：config `prompt_caching` 显式优先；否则按 base_url
/// 默认——`api.anthropic.com` → Full（满配），其余 → LastBreakpointOnly（保守）。
/// `base_url` 为 None 时按保守处理。
pub fn resolve_prompt_caching_policy(
    prompt_caching: Option<&str>,
    base_url: Option<&str>,
) -> PromptCachingPolicy {
    if let Some(s) = prompt_caching
        && let Some(p) = PromptCachingPolicy::parse(s)
    {
        return p;
    }
    match base_url
        .and_then(|u| url::Url::parse(u).ok())
        .and_then(|u| u.host_str().map(str::to_string))
        .as_deref()
    {
        Some("api.anthropic.com") => PromptCachingPolicy::Full,
        _ => PromptCachingPolicy::LastBreakpointOnly,
    }
}

/// 在请求上按 policy 放置 cache_control 断点。调用方在构造完请求后、发送前调用。
pub fn apply_prompt_caching(request: &mut AnthropicMessagesRequest, policy: PromptCachingPolicy) {
    let cc = ephemeral_cache_control();
    let mut used = 0usize;

    // ① system 末块（仅 Full）：Text 升级为 Blocks，标缓存。
    if policy == PromptCachingPolicy::Full
        && used < MAX_CACHE_BREAKPOINTS
        && let Some(system) = request.system.as_mut()
    {
        match system {
            AnthropicSystemPrompt::Text(text) => {
                *system = AnthropicSystemPrompt::Blocks(vec![AnthropicTextBlockParam {
                    text: std::mem::take(text),
                    cache_control: Some(cc.clone()),
                }]);
                used += 1;
            }
            AnthropicSystemPrompt::Blocks(blocks) => {
                if let Some(last) = blocks.last_mut() {
                    last.cache_control = Some(cc.clone());
                    used += 1;
                }
            }
        }
    }

    // ② 最后一个 tool（Full 和 LastBreakpointOnly 都标）。缓存其前的 system+tools 前缀。
    if used < MAX_CACHE_BREAKPOINTS
        && policy != PromptCachingPolicy::None
        && !request.tools.is_empty()
        && let Some(Value::Object(map)) = request.tools.last_mut()
        && let Ok(cc_val) = serde_json::to_value(&cc)
    {
        map.insert("cache_control".to_string(), cc_val);
        used += 1;
    }

    // ③④ messages[-2]、messages[-1] 末 text 块（仅 Full）。
    if policy == PromptCachingPolicy::Full {
        let len = request.messages.len();
        let start = len.saturating_sub(2);
        for i in start..len {
            if used >= MAX_CACHE_BREAKPOINTS {
                break;
            }
            if set_cache_on_message(&mut request.messages[i], cc.clone()) {
                used += 1;
            }
        }
    }

    enforce_cache_control_limit(request, MAX_CACHE_BREAKPOINTS);
}

/// 给一个 message 的末 text 块标缓存。`Text(s)` content 升级为 `Blocks`；
/// `Blocks` 委托 `set_cache_on_last_text_block`；无 text 块返回 false。
fn set_cache_on_message(msg: &mut AnthropicMessageParam, cc: AnthropicCacheControl) -> bool {
    match &mut msg.content {
        AnthropicMessageContent::Text(text) => {
            let text = std::mem::take(text);
            msg.content = AnthropicMessageContent::Blocks(vec![AnthropicContentBlockParam::Text {
                text,
                cache_control: Some(cc),
            }]);
            true
        }
        AnthropicMessageContent::Blocks(blocks) => set_cache_on_last_text_block(blocks, cc),
    }
}

/// 从尾往前找第一个 `Text`/`Image` 变体（有 `cache_control` 字段），置 `Some(cc)`。
fn set_cache_on_last_text_block(
    blocks: &mut [AnthropicContentBlockParam],
    cc: AnthropicCacheControl,
) -> bool {
    for block in blocks.iter_mut().rev() {
        match block {
            AnthropicContentBlockParam::Text { cache_control, .. }
            | AnthropicContentBlockParam::Image { cache_control, .. } => {
                *cache_control = Some(cc);
                return true;
            }
            AnthropicContentBlockParam::ToolUse { .. }
            | AnthropicContentBlockParam::ToolResult { .. } => continue,
        }
    }
    false
}

/// 数请求里所有 `Some(cache_control)`（system blocks + tools + messages blocks）。
fn count_breakpoints(request: &AnthropicMessagesRequest) -> usize {
    let mut n = 0usize;
    if let Some(AnthropicSystemPrompt::Blocks(blocks)) = request.system.as_ref() {
        n += blocks.iter().filter(|b| b.cache_control.is_some()).count();
    }
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
/// 再清 system 非末块、tools 非末块。正常路径为 no-op。
fn enforce_cache_control_limit(request: &mut AnthropicMessagesRequest, max: usize) {
    while count_breakpoints(request) > max {
        if let Some(i) = first_message_breakpoint_index(&request.messages) {
            clear_one_message_breakpoint(&mut request.messages[i]);
            continue;
        }
        if let Some(AnthropicSystemPrompt::Blocks(blocks)) = request.system.as_mut()
            && clear_non_last_block_breakpoint(blocks)
        {
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

fn clear_non_last_block_breakpoint(blocks: &mut [AnthropicTextBlockParam]) -> bool {
    let last_idx = blocks.len().saturating_sub(1);
    for (i, b) in blocks.iter_mut().enumerate() {
        if i != last_idx && b.cache_control.is_some() {
            b.cache_control = None;
            return true;
        }
    }
    false
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
    fn full_places_4_breakpoints() {
        let system = Some(AnthropicSystemPrompt::Text("sys".into()));
        let tools = vec![
            json!({"name":"a","input_schema":{}}),
            json!({"name":"b","input_schema":{}}),
        ];
        let messages = vec![user_text_msg("first"), user_text_msg("second")];
        let mut req = request_with(system, tools, messages);
        apply_prompt_caching(&mut req, PromptCachingPolicy::Full);

        assert!(matches!(
            req.system.as_ref().unwrap(),
            AnthropicSystemPrompt::Blocks(_)
        ));
        assert!(matches!(req.tools.last(), Some(Value::Object(m)) if m.contains_key("cache_control")));
        for m in &req.messages {
            match &m.content {
                AnthropicMessageContent::Blocks(b) => assert!(block_has_cc(b)),
                _ => panic!("content 应升级为 Blocks"),
            }
        }
        assert_eq!(count_breakpoints(&req), 4);
    }

    #[test]
    fn last_breakpoint_only_places_tool_only() {
        let system = Some(AnthropicSystemPrompt::Text("sys".into()));
        let tools = vec![json!({"name":"a","input_schema":{}}), json!({"name":"b","input_schema":{}})];
        let messages = vec![user_text_msg("first"), user_text_msg("second")];
        let mut req = request_with(system, tools, messages);
        apply_prompt_caching(&mut req, PromptCachingPolicy::LastBreakpointOnly);

        // system 仍 Text，messages 不动，只最后一个 tool 带 cc。
        assert!(matches!(req.system.as_ref().unwrap(), AnthropicSystemPrompt::Text(_)));
        for m in &req.messages {
            assert!(matches!(m.content, AnthropicMessageContent::Text(_)));
        }
        assert!(matches!(req.tools.last(), Some(Value::Object(m)) if m.contains_key("cache_control")));
        assert!(!matches!(req.tools.first(), Some(Value::Object(m)) if m.contains_key("cache_control")));
        assert_eq!(count_breakpoints(&req), 1);
    }

    #[test]
    fn none_leaves_request_untouched() {
        let mut req = request_with(
            Some(AnthropicSystemPrompt::Text("sys".into())),
            vec![json!({"name":"a","input_schema":{}})],
            vec![user_text_msg("hi")],
        );
        apply_prompt_caching(&mut req, PromptCachingPolicy::None);
        assert_eq!(count_breakpoints(&req), 0);
    }

    #[test]
    fn resolve_policy_config_overrides_base_url() {
        // config 显式优先于 base_url 启发式。
        assert_eq!(
            resolve_prompt_caching_policy(Some("full"), Some("https://dashscope/v1")),
            PromptCachingPolicy::Full
        );
        assert_eq!(
            resolve_prompt_caching_policy(Some("none"), Some("https://api.anthropic.com")),
            PromptCachingPolicy::None
        );
    }

    #[test]
    fn resolve_policy_default_by_base_url() {
        // 无 config：api.anthropic.com → Full，其余 → LastBreakpointOnly。
        assert_eq!(
            resolve_prompt_caching_policy(None, Some("https://api.anthropic.com")),
            PromptCachingPolicy::Full
        );
        assert_eq!(
            resolve_prompt_caching_policy(None, Some("https://dashscope.aliyuncs.com/apps/anthropic")),
            PromptCachingPolicy::LastBreakpointOnly
        );
        assert_eq!(
            resolve_prompt_caching_policy(None, None),
            PromptCachingPolicy::LastBreakpointOnly
        );
    }

    #[test]
    fn resolve_policy_unknown_config_falls_back_to_base_url() {
        // config 值无法识别时回落到 base_url 启发式。
        assert_eq!(
            resolve_prompt_caching_policy(Some("garbage"), Some("https://api.anthropic.com")),
            PromptCachingPolicy::Full
        );
    }

    #[test]
    fn enforces_4_cap_when_upstream_pre_seeded() {
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
        apply_prompt_caching(&mut req, PromptCachingPolicy::Full);
        assert!(count_breakpoints(&req) <= MAX_CACHE_BREAKPOINTS);
    }

    #[test]
    fn ephemeral_cc_is_short_no_ttl() {
        let cc = ephemeral_cache_control();
        assert_eq!(cc.cache_type, "ephemeral");
        assert!(cc.ttl.is_none());
        assert_eq!(serde_json::to_value(&cc).unwrap(), json!({"type":"ephemeral"}));
    }
}
