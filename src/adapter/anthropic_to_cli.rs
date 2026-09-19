//! Anthropic Messages request → `Conversation`.

use serde_json::Value;

use crate::conversation::{Block, Conversation, ConversationBuilder, ImageSource, Role};
use crate::types::anthropic::{ContentBlockInput, ContentInput, MessagesRequest};

pub fn to_conversation(request: &MessagesRequest) -> Result<Conversation, String> {
    let mut builder = ConversationBuilder::new();
    if let Some(system) = &request.system {
        let blocks = content_blocks(system, Role::User)?;
        let text: Vec<&str> = blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text(t) => Some(t.as_str()),
                Block::Image(_) => None,
            })
            .collect();
        builder.system(&text.join("\n"));
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
            "tool_result" => out.extend(tool_result_text(block).map(Block::Text)),
            // Thinking and tool calls from earlier assistant turns are not replayed.
            _ if role == Role::Assistant => {}
            other => return Err(format!("content block type '{other}' is not supported")),
        }
    }
    Ok(out)
}

fn image_source(block: &ContentBlockInput) -> Result<ImageSource, String> {
    let source = block.source.as_ref().ok_or("image block without a source")?;
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

/// A tool result's content is a string or a list of text blocks.
fn tool_result_text(block: &ContentBlockInput) -> Option<String> {
    match block.content.as_ref()? {
        Value::String(s) => Some(s.clone()),
        Value::Array(items) => Some(
            items
                .iter()
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn assistant_thinking_and_tool_use_are_skipped() {
        let c = conversation(
            r#"{"messages":[
                {"role":"user","content":"weather?"},
                {"role":"assistant","content":[{"type":"thinking","thinking":"hm"},{"type":"text","text":"Checking."},{"type":"tool_use","id":"t","name":"w","input":{}}]},
                {"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"sunny"}]}]}
            ]}"#,
        )
        .unwrap();
        assert_eq!(c.history()[1].blocks, vec![Block::Text("Checking.".into())]);
        assert_eq!(c.last().blocks, vec![Block::Text("sunny".into())]);
    }

    #[test]
    fn rejects_bad_requests() {
        assert!(conversation(r#"{"messages":[]}"#).is_err());
        assert!(conversation(r#"{"messages":[{"role":"system","content":"x"}]}"#).is_err());
        assert!(conversation(r#"{"messages":[{"role":"user","content":[{"type":"image","source":{"type":"file","file_id":"f"}}]}]}"#).is_err());
        assert!(conversation(r#"{"messages":[{"role":"user","content":[{"type":"document","source":{}}]}]}"#).is_err());
    }
}
