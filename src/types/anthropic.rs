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
    /// Client-side tools, which the CLI cannot call; ignored and logged.
    pub tools: Option<Value>,
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
/// are declared: `text`, `image` (`source`) and `tool_result` (`content`).
#[derive(Debug, Deserialize)]
pub struct ContentBlockInput {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: Option<String>,
    pub source: Option<Value>,
    pub content: Option<Value>,
}

// ── Response ───────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<TextBlock>,
    pub model: String,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct TextBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
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
            content: vec![TextBlock { block_type: "text".into(), text: "Hi".into() }],
            model: "claude-haiku-4-5-20251001".into(),
            stop_reason: "end_turn".into(),
            stop_sequence: None,
            usage: Usage { input_tokens: 5, output_tokens: 1, ..Usage::default() },
        };
        let v = serde_json::to_value(&response).unwrap();
        assert_eq!(v["type"], "message");
        assert_eq!(v["content"][0]["type"], "text");
        assert_eq!(v["usage"]["cache_read_input_tokens"], 0);
    }
}
