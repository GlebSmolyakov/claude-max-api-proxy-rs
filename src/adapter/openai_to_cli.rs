//! OpenAI chat request → `Conversation`.

use serde_json::{Value, json};

use crate::conversation::{Block, Conversation, ConversationBuilder, ImageSource, Role, ToolCall, ToolDef};
use crate::types::openai::{ChatCompletionRequest, MessageContent, ToolSpec};

pub fn to_conversation(request: &ChatCompletionRequest) -> Result<Conversation, String> {
    let messages = request
        .messages
        .as_deref()
        .filter(|m| !m.is_empty())
        .ok_or("messages is required and must be a non-empty array")?;

    let mut builder = ConversationBuilder::new();
    let tools_off = request.tool_choice.as_ref().and_then(Value::as_str) == Some("none");
    if let (Some(tools), false) = (&request.tools, tools_off) {
        builder.tools(tool_defs(tools));
    }

    for message in messages {
        let role = message.role.as_str();
        match role {
            // `developer` is the newer name for `system`.
            "system" | "developer" => {
                let blocks = content_blocks(message.content.as_ref(), role)?;
                builder.system(&text_of(&blocks));
            }
            "assistant" => {
                let mut blocks: Vec<Block> = content_blocks(message.content.as_ref(), role)?
                    .into_iter()
                    .filter(|b| matches!(b, Block::Text(_)))
                    .collect();
                for call in message.tool_calls.iter().flatten() {
                    blocks.push(Block::ToolUse(ToolCall {
                        id: call.id.clone(),
                        name: call.function.name.clone(),
                        input: parse_arguments(&call.function.arguments),
                    }));
                }
                builder.push(Role::Assistant, blocks);
            }
            "tool" if message.tool_call_id.is_some() => {
                let content = content_blocks(message.content.as_ref(), role)?;
                builder.push(
                    Role::User,
                    vec![Block::ToolResult {
                        tool_use_id: message.tool_call_id.clone().unwrap_or_default(),
                        content,
                        is_error: false,
                    }],
                );
            }
            // `user`, and the legacy `function` role, which reads best as user text.
            _ => builder.push(Role::User, content_blocks(message.content.as_ref(), role)?),
        }
    }
    builder.build()
}

fn tool_defs(tools: &[ToolSpec]) -> Vec<ToolDef> {
    tools
        .iter()
        .filter(|t| t.tool_type == "function")
        .filter_map(|t| t.function.as_ref())
        .map(|f| ToolDef {
            name: f.name.clone(),
            description: f.description.clone().unwrap_or_default(),
            input_schema: f
                .parameters
                .clone()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
        })
        .collect()
}

/// Arguments arrive as a JSON string; a string that is not JSON is kept as is.
fn parse_arguments(arguments: &Value) -> Value {
    match arguments {
        Value::String(s) if s.trim().is_empty() => json!({}),
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
        Value::Null => json!({}),
        other => other.clone(),
    }
}

fn content_blocks(content: Option<&MessageContent>, role: &str) -> Result<Vec<Block>, String> {
    let Some(content) = content else { return Ok(vec![]) };
    let parts = match content {
        MessageContent::Text(text) => return Ok(vec![Block::Text(text.clone())]),
        MessageContent::Parts(parts) => parts,
    };

    let mut blocks = Vec::new();
    for part in parts {
        match (part.part_type.as_str(), &part.text, &part.image_url) {
            ("text", Some(text), _) => blocks.push(Block::Text(text.clone())),
            ("image_url", _, Some(image)) => blocks.push(Block::Image(parse_image_url(image.url())?)),
            // Assistant parts like `refusal` carry nothing to replay.
            _ if role == "assistant" => {}
            (other, _, _) => return Err(format!("content part type '{other}' is not supported")),
        }
    }
    Ok(blocks)
}

fn text_of(blocks: &[Block]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `data:image/png;base64,...` becomes an inline image; `http(s)://` URLs are
/// passed on for the API to download.
pub fn parse_image_url(url: &str) -> Result<ImageSource, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (meta, data) = rest.split_once(',').ok_or("malformed image data URL")?;
        let media_type = meta
            .strip_suffix(";base64")
            .ok_or("image data URLs must be base64-encoded")?;
        if !media_type.starts_with("image/") {
            return Err(format!("unsupported image media type '{media_type}'"));
        }
        return Ok(ImageSource::Base64 {
            media_type: media_type.to_string(),
            data: data.to_string(),
        });
    }
    if url.starts_with("https://") || url.starts_with("http://") {
        return Ok(ImageSource::Url(url.to_string()));
    }
    Err("image_url must be a data: URL or an http(s) URL".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation(json: &str) -> Result<Conversation, String> {
        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        to_conversation(&request)
    }

    const READ_FILE: &str = r#"{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}"#;

    #[test]
    fn simple_message() {
        let c = conversation(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(c.system, "");
        assert!(c.tools.is_empty());
        assert_eq!(c.last().blocks, vec![Block::Text("hi".into())]);
    }

    #[test]
    fn system_and_developer_become_the_system_prompt() {
        let c = conversation(
            r#"{"messages":[{"role":"system","content":"Be brief."},{"role":"developer","content":[{"type":"text","text":"Use Russian."}]},{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(c.system, "Be brief.\n\nUse Russian.");
        assert_eq!(c.turns().len(), 1);
    }

    #[test]
    fn history_is_kept_as_turns() {
        let c = conversation(
            r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"more"}]}"#,
        )
        .unwrap();
        assert_eq!(c.turns().len(), 3);
        assert_eq!(c.history()[1].role, Role::Assistant);
    }

    #[test]
    fn images_become_blocks() {
        let c = conversation(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"what?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,QUJD"}},{"type":"image_url","image_url":{"url":"https://x/y.jpg"}}]}]}"#,
        )
        .unwrap();
        assert_eq!(
            c.last().blocks[1],
            Block::Image(ImageSource::Base64 { media_type: "image/png".into(), data: "QUJD".into() })
        );
        assert_eq!(c.last().blocks[2], Block::Image(ImageSource::Url("https://x/y.jpg".into())));
    }

    #[test]
    fn tools_are_declared_unless_tool_choice_is_none() {
        let with = conversation(&format!(r#"{{"tools":[{READ_FILE}],"messages":[{{"role":"user","content":"hi"}}]}}"#)).unwrap();
        assert_eq!(with.tools[0].name, "read_file");
        assert_eq!(with.tools[0].input_schema["properties"]["path"]["type"], "string");
        let none = conversation(&format!(r#"{{"tools":[{READ_FILE}],"tool_choice":"none","messages":[{{"role":"user","content":"hi"}}]}}"#)).unwrap();
        assert!(none.tools.is_empty());
    }

    #[test]
    fn a_tool_without_parameters_gets_an_empty_object_schema() {
        let c = conversation(r#"{"tools":[{"type":"function","function":{"name":"now"}}],"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(c.tools[0].input_schema, json!({ "type": "object", "properties": {} }));
    }

    #[test]
    fn tool_calls_and_results_become_blocks() {
        let c = conversation(
            r#"{"messages":[
                {"role":"user","content":"what is in a.rs?"},
                {"role":"assistant","content":"Let me look.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]},
                {"role":"tool","tool_call_id":"call_1","content":"fn main() {}"}
            ]}"#,
        )
        .unwrap();
        assert_eq!(
            c.history()[1].blocks,
            vec![
                Block::Text("Let me look.".into()),
                Block::ToolUse(ToolCall { id: "call_1".into(), name: "read_file".into(), input: json!({ "path": "a.rs" }) }),
            ]
        );
        let results = c.tool_results();
        assert_eq!(results[0].tool_use_id, "call_1");
        assert_eq!(results[0].content, vec![Block::Text("fn main() {}".into())]);
    }

    #[test]
    fn parallel_tool_results_share_one_turn() {
        let c = conversation(
            r#"{"messages":[
                {"role":"user","content":"read both"},
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}},
                    {"id":"b","type":"function","function":{"name":"read_file","arguments":""}}]},
                {"role":"tool","tool_call_id":"a","content":"1"},
                {"role":"tool","tool_call_id":"b","content":"2"}
            ]}"#,
        )
        .unwrap();
        assert_eq!(c.turns().len(), 3);
        assert_eq!(c.tool_results().len(), 2);
    }

    #[test]
    fn arguments_that_are_not_json_survive_as_a_string() {
        assert_eq!(parse_arguments(&json!("{\"a\":1}")), json!({ "a": 1 }));
        assert_eq!(parse_arguments(&json!("not json")), json!("not json"));
        assert_eq!(parse_arguments(&json!({ "a": 1 })), json!({ "a": 1 }));
        assert_eq!(parse_arguments(&Value::Null), json!({}));
    }

    #[test]
    fn rejects_bad_requests() {
        assert!(conversation(r#"{"messages":[]}"#).is_err());
        assert!(conversation(r#"{}"#).is_err());
        assert!(conversation(r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#).is_err());
        let audio = conversation(r#"{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{}}]}]}"#);
        assert_eq!(audio.unwrap_err(), "content part type 'input_audio' is not supported");
    }

    #[test]
    fn image_url_parsing() {
        assert!(parse_image_url("data:image/png;base64,AAAA").is_ok());
        assert!(parse_image_url("data:image/png,AAAA").is_err(), "not base64");
        assert!(parse_image_url("data:text/plain;base64,AAAA").is_err());
        assert!(parse_image_url("data:image/png;base64").is_err(), "no comma");
        assert!(parse_image_url("file:///etc/passwd").is_err());
    }
}
