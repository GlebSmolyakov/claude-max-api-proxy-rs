//! Anthropic Messages request → `Conversation`.

use serde_json::Value;

use crate::conversation::{Block, Conversation, ConversationBuilder, ImageSource, Role, ToolCall, ToolDef};
use crate::types::anthropic::{ContentBlockInput, ContentInput, MessagesRequest};

pub fn to_conversation(request: &MessagesRequest) -> Result<Conversation, String> {
    let mut builder = ConversationBuilder::new();
    if let Some(system) = &request.system {
        let blocks = content_blocks(system, Role::User)?;
        let text: Vec<&str> = blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        builder.system(&text.join("\n"));
    }

    let tools_off = request
        .tool_choice
        .as_ref()
        .and_then(|c| c.get("type"))
        .and_then(Value::as_str)
        == Some("none");
    if let (Some(tools), false) = (&request.tools, tools_off) {
        // Server tools (web search and the like) have no input schema and
        // run on Anthropic's side; only client tools go to the bridge.
        let defs = tools
            .iter()
            .filter_map(|t| {
                Some(ToolDef {
                    name: t.name.clone(),
                    description: t.description.clone().unwrap_or_default(),
                    input_schema: t.input_schema.clone()?,
                })
            })
            .collect();
        builder.tools(defs);
    }

    for message in &request.messages {
        let role = match message.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => return Err(format!("unsupported message role '{other}'")),
        };
        builder.push(role, content_blocks(&message.content, role)?);
    }
    builder.build()
}

fn content_blocks(content: &ContentInput, role: Role) -> Result<Vec<Block>, String> {
    let blocks = match content {
        ContentInput::Text(text) => return Ok(vec![Block::Text(text.clone())]),
        ContentInput::Blocks(blocks) => blocks,
    };
    let mut out = Vec::new();
    for block in blocks {
        match block.block_type.as_str() {
            "text" => out.extend(block.text.clone().map(Block::Text)),
            "image" if role == Role::User => out.push(Block::Image(image_source(block)?)),
            "tool_use" if role == Role::Assistant => out.push(Block::ToolUse(ToolCall {
                id: block.id.clone().ok_or("tool_use block without an id")?,
                name: block.name.clone().ok_or("tool_use block without a name")?,
                input: block.input.clone().unwrap_or_else(|| serde_json::json!({})),
            })),
            "tool_result" if role == Role::User => out.push(Block::ToolResult {
                tool_use_id: block.tool_use_id.clone().ok_or("tool_result block without a tool_use_id")?,
                content: tool_result_content(block.content.as_ref())?,
                is_error: block.is_error,
            }),
            // Thinking from earlier assistant turns is not replayed.
            _ if role == Role::Assistant => {}
            other => return Err(format!("content block type '{other}' is not supported")),
        }
    }
    Ok(out)
}

fn image_source(block: &ContentBlockInput) -> Result<ImageSource, String> {
    image_from(block.source.as_ref().ok_or("image block without a source")?)
}

fn image_from(source: &Value) -> Result<ImageSource, String> {
    let field = |name: &str| source.get(name).and_then(Value::as_str).map(str::to_string);
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => Ok(ImageSource::Base64 {
            media_type: field("media_type").ok_or("base64 image without media_type")?,
            data: field("data").ok_or("base64 image without data")?,
        }),
        Some("url") => Ok(ImageSource::Url(field("url").ok_or("url image without url")?)),
        other => Err(format!("unsupported image source type {other:?}")),
    }
}

/// A tool result's content is a string or a list of text and image blocks.
fn tool_result_content(content: Option<&Value>) -> Result<Vec<Block>, String> {
    match content {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::String(s)) => Ok(vec![Block::Text(s.clone())]),
        Some(Value::Array(items)) => {
            let mut out = Vec::new();
            for item in items {
                match item.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        out.extend(item.get("text").and_then(Value::as_str).map(|t| Block::Text(t.to_string())))
                    }
                    Some("image") => out.push(Block::Image(image_from(
                        item.get("source").ok_or("image block without a source")?,
                    )?)),
                    _ => {}
                }
            }
            Ok(out)
        }
        Some(other) => Ok(vec![Block::Text(other.to_string())]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn conversation(json: &str) -> Result<Conversation, String> {
        let request: MessagesRequest = serde_json::from_str(json).unwrap();
        to_conversation(&request)
    }

    #[test]
    fn string_system_and_message() {
        let c = conversation(r#"{"system":"Be brief.","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(c.system, "Be brief.");
        assert_eq!(c.last().blocks, vec![Block::Text("hi".into())]);
    }

    #[test]
    fn system_blocks_are_joined() {
        let c = conversation(
            r#"{"system":[{"type":"text","text":"One."},{"type":"text","text":"Two."}],"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(c.system, "One.\nTwo.");
    }

    #[test]
    fn images_in_both_source_shapes() {
        let c = conversation(
            r#"{"messages":[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"QUJD"}},
                {"type":"image","source":{"type":"url","url":"https://x/y.png"}},
                {"type":"text","text":"compare"}
            ]}]}"#,
        )
        .unwrap();
        assert_eq!(
            c.last().blocks[0],
            Block::Image(ImageSource::Base64 { media_type: "image/png".into(), data: "QUJD".into() })
        );
        assert_eq!(c.last().blocks[1], Block::Image(ImageSource::Url("https://x/y.png".into())));
    }

    #[test]
    fn client_tools_are_declared_and_server_tools_skipped() {
        let c = conversation(
            r#"{"tools":[{"name":"read_file","description":"Read","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search"}],
                "messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert_eq!(c.tools.len(), 1);
        assert_eq!(c.tools[0].name, "read_file");
        let none = conversation(
            r#"{"tools":[{"name":"read_file","input_schema":{"type":"object"}}],"tool_choice":{"type":"none"},"messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        assert!(none.tools.is_empty());
    }

    #[test]
    fn tool_use_and_tool_result_become_blocks() {
        let c = conversation(
            r#"{"messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":[{"type":"thinking","thinking":"hm"},{"type":"text","text":"Checking."},{"type":"tool_use","id":"t","name":"weather","input":{"city":"Lisbon"}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"sunny"}],"is_error":false}]}
            ]}"#,
        )
        .unwrap();
        assert_eq!(
            c.history()[1].blocks,
            vec![
                Block::Text("Checking.".into()),
                Block::ToolUse(ToolCall { id: "t".into(), name: "weather".into(), input: json!({ "city": "Lisbon" }) }),
            ]
        );
        let results = c.tool_results();
        assert_eq!(results[0].tool_use_id, "t");
        assert_eq!(results[0].content, vec![Block::Text("sunny".into())]);
        assert!(!results[0].is_error);
    }

    #[test]
    fn tool_results_carry_errors_strings_and_images() {
        let c = conversation(
            r#"{"messages":[{"role":"user","content":[
                {"type":"tool_result","tool_use_id":"a","content":"no such file","is_error":true},
                {"type":"tool_result","tool_use_id":"b","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"QQ=="}}]}
            ]}]}"#,
        )
        .unwrap();
        let results = c.tool_results();
        assert!(results[0].is_error);
        assert_eq!(results[0].content, vec![Block::Text("no such file".into())]);
        assert!(matches!(results[1].content[0], Block::Image(ImageSource::Base64 { .. })));
    }

    #[test]
    fn rejects_bad_requests() {
        assert!(conversation(r#"{"messages":[]}"#).is_err());
        assert!(conversation(r#"{"messages":[{"role":"system","content":"x"}]}"#).is_err());
        assert!(conversation(r#"{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"file","file_id":"f"}}]}]}"#).is_err());
        assert!(conversation(r#"{"messages":[{"role":"user","content":[{"type":"document","source":{}}]}]}"#).is_err());
        assert!(conversation(r#"{"messages":[{"role":"user","content":[{"type":"tool_result","content":"x"}]}]}"#).is_err(), "no tool_use_id");
    }
}
