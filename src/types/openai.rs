use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request ────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Message>>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    /// Function tools the client can run.
    pub tools: Option<Vec<ToolSpec>>,
    /// `"none"` hides the tools from the model; other values are not enforced.
    pub tool_choice: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: Option<FunctionSpec>,
}

#[derive(Debug, Deserialize)]
pub struct FunctionSpec {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema of the arguments.
    pub parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<MessageContent>,
    /// On assistant messages: tools the model called.
    pub tool_calls: Option<Vec<MessageToolCall>>,
    /// On `tool` messages: the call this result answers.
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MessageToolCall {
    pub id: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// A JSON string per the spec; some clients send the object itself.
    #[serde(default)]
    pub arguments: Value,
}

/// Message content is a plain string or an array of parts.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    pub text: Option<String>,
    pub image_url: Option<ImageUrl>,
}

/// `{"url": "..."}` per the spec; some clients send the URL as a bare string.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ImageUrl {
    Bare(String),
    Object { url: String },
}

impl ImageUrl {
    pub fn url(&self) -> &str {
        match self {
            ImageUrl::Bare(url) | ImageUrl::Object { url } => url,
        }
    }
}

// ── Response ───────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    /// `null` when the model only called tools.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ResponseFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseFunction {
    pub name: String,
    /// The arguments as a JSON string.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Usage {
    /// All input tokens, cached or not, as OpenAI counts them.
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Only on the extra last chunk requested with `stream_options.include_usage`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChunkToolCall>>,
}

/// A tool call in a stream chunk; this proxy sends each one whole.
#[derive(Debug, Serialize)]
pub struct ChunkToolCall {
    pub index: u32,
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ResponseFunction,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub created: u64,
    /// Known once a turn has used the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_content() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"haiku","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        let msg = &req.messages.unwrap()[0];
        assert!(matches!(&msg.content, Some(MessageContent::Text(t)) if t == "hi"));
        assert!(!req.stream);
    }

    #[test]
    fn image_parts_in_both_shapes() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"what?"},
                {"type":"image_url","image_url":{"url":"https://x/a.png","detail":"high"}},
                {"type":"image_url","image_url":"https://x/b.png"}
            ]}]}"#,
        )
        .unwrap();
        let Some(MessageContent::Parts(parts)) = &req.messages.as_ref().unwrap()[0].content else { panic!() };
        assert_eq!(parts[1].image_url.as_ref().unwrap().url(), "https://x/a.png");
        assert_eq!(parts[2].image_url.as_ref().unwrap().url(), "https://x/b.png");
    }

    #[test]
    fn stream_options_and_tools() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[],"stream":true,"stream_options":{"include_usage":true},"tool_choice":"auto",
                "tools":[{"type":"function","function":{"name":"read_file","description":"Read","parameters":{"type":"object"}}}]}"#,
        )
        .unwrap();
        assert!(req.stream);
        assert!(req.stream_options.unwrap().include_usage);
        let tools = req.tools.unwrap();
        assert_eq!(tools[0].function.as_ref().unwrap().name, "read_file");
        assert_eq!(req.tool_choice.unwrap(), "auto");
    }

    #[test]
    fn tool_calls_and_tool_messages() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[
                {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"fn main() {}"}
            ]}"#,
        )
        .unwrap();
        let messages = req.messages.unwrap();
        let call = &messages[0].tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.function.arguments, "{\"path\":\"a.rs\"}");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn null_content_is_accepted() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[{"role":"assistant","content":null,"tool_calls":[]}]}"#).unwrap();
        assert!(req.messages.unwrap()[0].content.is_none());
    }

    #[test]
    fn chunk_without_usage_omits_the_field() {
        let chunk = ChatCompletionChunk {
            id: "c".into(),
            object: "chat.completion.chunk".into(),
            created: 1,
            model: "m".into(),
            choices: vec![],
            usage: None,
        };
        assert!(serde_json::to_value(&chunk).unwrap().get("usage").is_none());
    }

    #[test]
    fn tool_only_message_has_null_content() {
        let message = ResponseMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![ResponseToolCall {
                id: "toolu_1".into(),
                call_type: "function".into(),
                function: ResponseFunction { name: "read_file".into(), arguments: "{}".into() },
            }]),
        };
        let v = serde_json::to_value(&message).unwrap();
        assert!(v["content"].is_null());
        assert_eq!(v["tool_calls"][0]["type"], "function");
    }

    #[test]
    fn model_info_omits_unknown_limits() {
        let info = ModelInfo {
            id: "haiku".into(),
            object: "model".into(),
            owned_by: "anthropic".into(),
            created: 1,
            context_window: None,
            max_tokens: Some(32_000),
        };
        let v = serde_json::to_value(&info).unwrap();
        assert!(v.get("context_window").is_none());
        assert_eq!(v["max_tokens"], 32_000);
    }
}
