//! OpenAI chat request → `Conversation`.

use crate::conversation::{Block, Conversation, ConversationBuilder, ImageSource, Role};
use crate::types::openai::{ChatCompletionRequest, MessageContent};

pub fn to_conversation(request: &ChatCompletionRequest) -> Result<Conversation, String> {
    let messages = request
        .messages
        .as_deref()
        .filter(|m| !m.is_empty())
        .ok_or("messages is required and must be a non-empty array")?;

    let mut builder = ConversationBuilder::new();
    for message in messages {
        let role = message.role.as_str();
        match role {
            // `developer` is the newer name for `system`.
            "system" | "developer" => {
                let blocks = content_blocks(message.content.as_ref(), role)?;
                builder.system(&text_of(&blocks));
            }
            "assistant" => {
                let blocks = content_blocks(message.content.as_ref(), role)?;
                let text: Vec<Block> = blocks.into_iter().filter(|b| matches!(b, Block::Text(_))).collect();
                builder.push(Role::Assistant, text);
            }
            // `user`, and `tool`/`function` results, which read best as user text.
            _ => builder.push(Role::User, content_blocks(message.content.as_ref(), role)?),
        }
    }
    builder.build()
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
            Block::Image(_) => None,
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

    #[test]
    fn simple_message() {
        let c = conversation(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(c.system, "");
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
    fn tool_results_read_as_user_text() {
        let c = conversation(
            r#"{"messages":[{"role":"user","content":"weather?"},{"role":"assistant","content":null,"tool_calls":[{}]},{"role":"tool","content":"sunny"}]}"#,
        )
        .unwrap();
        assert_eq!(c.turns().len(), 1, "the empty assistant turn drops and the user texts merge");
        assert_eq!(c.last().blocks, vec![Block::Text("weather?".into()), Block::Text("sunny".into())]);
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
