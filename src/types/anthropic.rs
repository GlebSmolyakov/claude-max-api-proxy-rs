use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    /// `max_tokens`, `temperature` and other sampling fields are accepted and
    /// ignored: the CLI chooses them itself.
    pub model: Option<String>,
    pub messages: Vec<MessageInput>,
    #[serde(default)]
    pub stream: bool,
    pub system: Option<ContentInput>,
    pub tools: Option<Vec<ToolSpec>>,
    /// `{"type": "none"}` hides the tools from the model; other values are not enforced.
    pub tool_choice: Option<Value>,
}

/// A client tool. Server tools such as web search have a versioned `type`
/// and no `input_schema`; the proxy skips those.
#[derive(Debug, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct MessageInput {
    pub role: String,
    pub content: ContentInput,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ContentInput {
    Text(String),
    Blocks(Vec<ContentBlockInput>),
}

/// One content block. Only the fields of the block types the proxy reads
/// are declared: `text`, `image` (`source`), `tool_use` (`id`, `name`,
/// `input`) and `tool_result` (`tool_use_id`, `content`, `is_error`).
#[derive(Debug, Deserialize)]
pub struct ContentBlockInput {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
    pub source: Option<Value>,
    pub id: Option<String>,
    pub name: Option<String>,
    pub input: Option<Value>,
    pub tool_use_id: Option<String>,
    pub content: Option<Value>,
    #[serde(default)]
    pub is_error: bool,
}

// ── Response ───────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ResponseBlock>,
    pub model: String,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
}

#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_and_block_content() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"model":"haiku","max_tokens":100,"messages":[
                {"role":"user","content":"hi"},
                {"role":"assistant","content":[{"type":"text","text":"hello"}]},
                {"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"QQ=="}}]}
            ]}"#,
        )
        .unwrap();
        assert!(matches!(&req.messages[0].content, ContentInput::Text(t) if t == "hi"));
        let ContentInput::Blocks(blocks) = &req.messages[2].content else { panic!() };
        assert_eq!(blocks[0].source.as_ref().unwrap()["media_type"], "image/png");
    }

    #[test]
    fn tools_tool_use_and_tool_result() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"tools":[{"name":"read_file","description":"Read","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search"}],
                "messages":[
                {"role":"user","content":"go"},
                {"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"a.rs"}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"fn main() {}","is_error":false}]}
            ]}"#,
        )
        .unwrap();
        let tools = req.tools.unwrap();
        assert!(tools[0].input_schema.is_some());
        assert!(tools[1].input_schema.is_none(), "a server tool");
        let ContentInput::Blocks(call) = &req.messages[1].content else { panic!() };
        assert_eq!(call[0].name.as_deref(), Some("read_file"));
        let ContentInput::Blocks(result) = &req.messages[2].content else { panic!() };
        assert_eq!(result[0].tool_use_id.as_deref(), Some("toolu_1"));
    }

    #[test]
    fn model_is_optional() {
        let req: MessagesRequest = serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(req.model, None);
    }

    #[test]
    fn system_as_blocks() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"messages":[],"system":[{"type":"text","text":"Be brief.","cache_control":{"type":"ephemeral"}}]}"#,
        )
        .unwrap();
        assert!(matches!(req.system, Some(ContentInput::Blocks(_))));
    }

    #[test]
    fn response_shape() {
        let response = MessagesResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![
                ResponseBlock::Text { text: "Hi".into() },
                ResponseBlock::ToolUse { id: "toolu_1".into(), name: "read_file".into(), input: serde_json::json!({}) },
            ],
            model: "claude-haiku-4-5-20251001".into(),
            stop_reason: "end_turn".into(),
            stop_sequence: None,
            usage: Usage { input_tokens: 5, output_tokens: 1, ..Usage::default() },
        };
        let v = serde_json::to_value(&response).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["content"][1]["type"], "tool_use");
        assert_eq!(v["content"][1]["name"], "read_file");
        assert_eq!(v["usage"]["cache_read_input_tokens"], 0);
    }
}
