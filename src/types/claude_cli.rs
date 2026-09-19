//! Lines the CLI prints with `--output-format stream-json`.
//!
//! Only the fields the proxy uses are declared; serde ignores the rest, and
//! line types the proxy does not care about fail to parse and are skipped.

use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClaudeCliMessage {
    #[serde(rename = "system")]
    System(SystemMessage),

    /// One finished content block of the assistant message. Text also
    /// arrives as stream deltas, so only tool calls are read from here: they
    /// carry the complete input.
    #[serde(rename = "assistant")]
    Assistant { message: AssistantMessage },

    #[serde(rename = "stream_event")]
    StreamEvent { event: StreamEvent },

    #[serde(rename = "rate_limit_event")]
    RateLimit { rate_limit_info: RateLimitInfo },

    #[serde(rename = "result")]
    Result(ResultMessage),
}

/// `system` lines. The first one, `subtype: "init"`, names the session and
/// the model the alias resolved to.
#[derive(Debug, Deserialize)]
pub struct SystemMessage {
    pub subtype: Option<String>,
    pub model: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    #[serde(default)]
    pub content: Vec<AssistantBlock>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantBlock {
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },

    #[serde(other)]
    Other,
}

/// Raw Messages API stream events, forwarded by the CLI.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StreamEvent {
    #[serde(rename = "content_block_start")]
    ContentBlockStart { content_block: StartedBlock },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: Delta },

    /// The end of one API call: why it stopped and what it cost.
    #[serde(rename = "message_delta")]
    MessageDelta {
        #[serde(default)]
        delta: MessageDeltaInfo,
        usage: Option<ResultUsage>,
    },

    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StartedBlock {
    /// Announces a tool call before its input is complete.
    #[serde(rename = "tool_use")]
    ToolUse { id: String },

    #[serde(other)]
    Other,
}

#[derive(Debug, Default, Deserialize)]
pub struct MessageDeltaInfo {
    /// `end_turn`, `tool_use`, `max_tokens`, …
    pub stop_reason: Option<String>,
}

/// `text_delta` carries `text`; `thinking_delta` and `signature_delta` do
/// not, which keeps thinking out of the reply.
#[derive(Debug, Deserialize)]
pub struct Delta {
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitInfo {
    /// `"allowed"` while the subscription has room.
    pub status: Option<String>,
    /// Utilization per window, e.g. `five_hour` and `seven_day`.
    #[serde(rename = "unifiedWindows", default)]
    pub windows: HashMap<String, RateLimitWindow>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitWindow {
    pub utilization: Option<f64>,
    #[serde(rename = "resetsAt")]
    pub resets_at: Option<u64>,
}

/// The last line of every run.
#[derive(Debug, Deserialize)]
pub struct ResultMessage {
    pub subtype: Option<String>,
    #[serde(default)]
    pub is_error: bool,
    /// Final reply text, or the error text when `is_error` is set.
    pub result: Option<String>,
    pub session_id: Option<String>,
    pub stop_reason: Option<String>,
    /// HTTP status of the API error behind `is_error`, when there was one.
    pub api_error_status: Option<u16>,
    pub usage: Option<ResultUsage>,
    #[serde(rename = "modelUsage")]
    pub model_usage: Option<HashMap<String, ModelUsage>>,
}

/// Token counts in the Messages API's own snake_case shape.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq)]
pub struct ResultUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

/// Per-model facts from `modelUsage`, which, unlike `usage`, is camelCase.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelUsage {
    pub context_window: Option<u64>,
    pub max_output_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A result line captured from CLI 2.1.276, trimmed of unused fields.
    const SUCCESS: &str = r#"{"duration_api_ms":1257,"stop_reason":"end_turn","session_id":"18229ada-4fef-49fb-8a31-ee3f5098df58","total_cost_usd":0.000635,"usage":{"input_tokens":430,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":41,"output_tokens_details":{"thinking_tokens":35},"service_tier":"standard"},"modelUsage":{"claude-haiku-4-5-20251001":{"inputTokens":430,"outputTokens":41,"cacheReadInputTokens":0,"cacheCreationInputTokens":0,"costUSD":0.000635,"contextWindow":200000,"maxOutputTokens":32000}},"is_error":false,"num_turns":1,"subtype":"success","api_error_status":null,"result":"Hi","type":"result"}"#;

    /// What the CLI prints when `--resume` names a session it does not have.
    const MISSING_SESSION: &str = r#"{"type":"result","subtype":"error_during_execution","duration_ms":0,"is_error":true,"num_turns":0,"stop_reason":null,"session_id":"279437d3-7926-4022-ac8e-111c74117a16","usage":{"input_tokens":0,"output_tokens":0},"modelUsage":{}}"#;

    fn parse(line: &str) -> ClaudeCliMessage {
        serde_json::from_str(line).unwrap()
    }

    #[test]
    fn parses_success_result_with_real_usage() {
        let ClaudeCliMessage::Result(r) = parse(SUCCESS) else { panic!() };
        assert!(!r.is_error);
        assert_eq!(r.result.as_deref(), Some("Hi"));
        assert_eq!(r.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(r.session_id.as_deref(), Some("18229ada-4fef-49fb-8a31-ee3f5098df58"));
        let usage = r.usage.unwrap();
        assert_eq!((usage.input_tokens, usage.output_tokens), (430, 41));
        let facts = &r.model_usage.unwrap()["claude-haiku-4-5-20251001"];
        assert_eq!(facts.context_window, Some(200_000));
        assert_eq!(facts.max_output_tokens, Some(32_000));
    }

    #[test]
    fn parses_error_result() {
        let ClaudeCliMessage::Result(r) = parse(MISSING_SESSION) else { panic!() };
        assert!(r.is_error);
        assert_eq!(r.subtype.as_deref(), Some("error_during_execution"));
        assert_eq!(r.result, None);
    }

    #[test]
    fn parses_api_error_status() {
        let line = r#"{"type":"result","subtype":"success","is_error":true,"api_error_status":400,"result":"API Error: 400 Unable to download the file."}"#;
        let ClaudeCliMessage::Result(r) = parse(line) else { panic!() };
        assert_eq!(r.api_error_status, Some(400));
    }

    #[test]
    fn parses_init() {
        let line = r#"{"type":"system","subtype":"init","cwd":"/tmp","session_id":"abc","tools":[],"model":"claude-haiku-4-5-20251001"}"#;
        let ClaudeCliMessage::System(s) = parse(line) else { panic!() };
        assert_eq!(s.subtype.as_deref(), Some("init"));
        assert_eq!(s.model.as_deref(), Some("claude-haiku-4-5-20251001"));
        assert_eq!(s.session_id.as_deref(), Some("abc"));
    }

    #[test]
    fn parses_rate_limit_event() {
        let line = r#"{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","resetsAt":1789824000,"rateLimitType":"five_hour","unifiedWindows":{"five_hour":{"utilization":0.1,"resetsAt":1789824000},"seven_day":{"utilization":0.01,"resetsAt":1790334000}}}}"#;
        let ClaudeCliMessage::RateLimit { rate_limit_info: info } = parse(line) else { panic!() };
        assert_eq!(info.status.as_deref(), Some("allowed"));
        assert_eq!(info.windows["seven_day"].utilization, Some(0.01));
    }

    #[test]
    fn text_and_thinking_deltas() {
        let text = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}}"#;
        let ClaudeCliMessage::StreamEvent { event: StreamEvent::ContentBlockDelta { delta } } = parse(text) else { panic!() };
        assert_eq!(delta.text.as_deref(), Some("Hi"));

        let thinking = r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}}"#;
        let ClaudeCliMessage::StreamEvent { event: StreamEvent::ContentBlockDelta { delta } } = parse(thinking) else { panic!() };
        assert_eq!(delta.text, None);
    }

    #[test]
    fn other_stream_events_parse_as_ignorable() {
        let start = r#"{"type":"stream_event","event":{"type":"message_start","message":{"model":"x"}}}"#;
        assert!(matches!(parse(start), ClaudeCliMessage::StreamEvent { event: StreamEvent::Other }));
        let text_start = r#"{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}}"#;
        assert!(matches!(
            parse(text_start),
            ClaudeCliMessage::StreamEvent { event: StreamEvent::ContentBlockStart { content_block: StartedBlock::Other } }
        ));
    }

    #[test]
    fn assistant_text_blocks_are_other_and_tool_use_blocks_carry_input() {
        let text = r#"{"type":"assistant","message":{"model":"x","content":[{"type":"text","text":"Hi"}]}}"#;
        let ClaudeCliMessage::Assistant { message } = parse(text) else { panic!() };
        assert!(matches!(message.content[..], [AssistantBlock::Other]));

        // Captured from CLI 2.1.278 calling a tool through an MCP server.
        let tool = r#"{"type":"assistant","message":{"model":"claude-haiku-4-5-20251001","content":[{"type":"tool_use","id":"toolu_019vyESf3ToWrVN4jfgMTXtN","name":"mcp__c__get_weather","input":{"city":"Lisbon"},"caller":{"type":"direct"}}]}}"#;
        let ClaudeCliMessage::Assistant { message } = parse(tool) else { panic!() };
        let [AssistantBlock::ToolUse { id, name, input }] = &message.content[..] else { panic!() };
        assert_eq!(id, "toolu_019vyESf3ToWrVN4jfgMTXtN");
        assert_eq!(name, "mcp__c__get_weather");
        assert_eq!(input["city"], "Lisbon");
    }

    #[test]
    fn tool_use_block_start_and_step_end() {
        let start = r#"{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"mcp__c__x","input":{}}}}"#;
        let ClaudeCliMessage::StreamEvent { event: StreamEvent::ContentBlockStart { content_block: StartedBlock::ToolUse { id } } } = parse(start) else { panic!() };
        assert_eq!(id, "toolu_1");

        let end = r#"{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":1143,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":128}}}"#;
        let ClaudeCliMessage::StreamEvent { event: StreamEvent::MessageDelta { delta, usage } } = parse(end) else { panic!() };
        assert_eq!(delta.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(usage.unwrap().output_tokens, 128);
    }

    #[test]
    fn unknown_line_types_do_not_parse() {
        assert!(serde_json::from_str::<ClaudeCliMessage>(r#"{"type":"user","message":{}}"#).is_err());
    }
}
