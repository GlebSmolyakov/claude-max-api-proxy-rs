//! Provider-neutral view of a chat request.
//!
//! Both adapters (OpenAI and Anthropic) turn their request into a
//! `Conversation`: one system prompt plus alternating user and assistant
//! turns made of text and image blocks. From it the turn runner builds what
//! the CLI reads on stdin, and the session store derives the keys that tie a
//! conversation prefix to a saved CLI session.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Text(String),
    Image(ImageSource),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub role: Role,
    pub blocks: Vec<Block>,
}

/// A validated conversation: at least one turn, and the last turn is the
/// user's new message.
#[derive(Debug, Clone, PartialEq)]
pub struct Conversation {
    pub system: String,
    turns: Vec<Turn>,
}

/// Collects messages in request order. Consecutive messages with the same
/// role merge into one turn, because the CLI, like the Messages API, expects
/// user and assistant turns to alternate.
#[derive(Debug, Default)]
pub struct ConversationBuilder {
    system: Vec<String>,
    turns: Vec<Turn>,
}

impl ConversationBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn system(&mut self, text: &str) {
        if !text.trim().is_empty() {
            self.system.push(text.to_string());
        }
    }

    pub fn push(&mut self, role: Role, blocks: Vec<Block>) {
        let blocks: Vec<Block> = blocks.into_iter().filter(|b| !is_blank(b)).collect();
        if blocks.is_empty() {
            return;
        }
        match self.turns.last_mut() {
            Some(last) if last.role == role => last.blocks.extend(blocks),
            _ => self.turns.push(Turn { role, blocks }),
        }
    }

    pub fn build(self) -> Result<Conversation, String> {
        match self.turns.last() {
            None => Err("messages must contain at least one non-empty message".to_string()),
            Some(turn) if turn.role != Role::User => {
                Err("the last message must come from the user".to_string())
            }
            Some(_) => Ok(Conversation {
                system: self.system.join("\n\n"),
                turns: self.turns,
            }),
        }
    }
}

fn is_blank(block: &Block) -> bool {
    matches!(block, Block::Text(s) if s.trim().is_empty())
}

impl Conversation {
    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Everything before the new user message.
    pub fn history(&self) -> &[Turn] {
        &self.turns[..self.turns.len() - 1]
    }

    /// The new user message.
    pub fn last(&self) -> &Turn {
        self.turns.last().expect("a built conversation has at least one turn")
    }

    /// Key of the history, or `None` when this is the first message.
    pub fn history_key(&self) -> Option<String> {
        let history = self.history();
        if history.is_empty() {
            None
        } else {
            Some(key_of(&self.system, history.iter()))
        }
    }

    /// The key the next request's history will hash to once the client has
    /// received `reply` and sends it back as the assistant turn.
    pub fn key_after_reply(&self, reply: &str) -> String {
        let reply_turn = Turn {
            role: Role::Assistant,
            blocks: vec![Block::Text(reply.to_string())],
        };
        key_of(&self.system, self.turns.iter().chain(std::iter::once(&reply_turn)))
    }

    /// CLI input when a saved session already holds the history: only the new message.
    pub fn continuation_input(&self) -> String {
        cli_input_line(&self.last().blocks)
    }

    /// CLI input for a fresh session. Without history this is the message
    /// itself. With history, the earlier turns are replayed as a transcript
    /// inside one message, because the CLI cannot be handed prior assistant
    /// turns directly. Images keep their place in the transcript.
    pub fn fresh_input(&self) -> String {
        let history = self.history();
        if history.is_empty() {
            return cli_input_line(&self.last().blocks);
        }

        let mut blocks = Vec::new();
        push_text(&mut blocks, "The conversation so far, for context:\n\n<conversation_history>");
        for turn in history {
            let tag = match turn.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            push_text(&mut blocks, &format!("\n<{tag}>\n"));
            append_blocks(&mut blocks, &turn.blocks);
            push_text(&mut blocks, &format!("\n</{tag}>"));
        }
        push_text(&mut blocks, "\n</conversation_history>\n\nReply to the latest user message:\n\n");
        append_blocks(&mut blocks, &self.last().blocks);
        cli_input_line(&blocks)
    }
}

/// Append text to the last block when it is text, so the transcript stays
/// one text block between images.
fn push_text(blocks: &mut Vec<Block>, text: &str) {
    if let Some(Block::Text(prev)) = blocks.last_mut() {
        prev.push_str(text);
    } else {
        blocks.push(Block::Text(text.to_string()));
    }
}

fn append_blocks(blocks: &mut Vec<Block>, from: &[Block]) {
    for block in from {
        match block {
            Block::Text(s) => push_text(blocks, s),
            Block::Image(_) => blocks.push(block.clone()),
        }
    }
}

/// One NDJSON line for `claude --input-format stream-json`.
fn cli_input_line(blocks: &[Block]) -> String {
    let content: Vec<Value> = blocks.iter().map(block_json).collect();
    let line = json!({
        "type": "user",
        "message": { "role": "user", "content": content },
    });
    format!("{line}\n")
}

fn block_json(block: &Block) -> Value {
    match block {
        Block::Text(text) => json!({ "type": "text", "text": text }),
        Block::Image(ImageSource::Base64 { media_type, data }) => json!({
            "type": "image",
            "source": { "type": "base64", "media_type": media_type, "data": data },
        }),
        Block::Image(ImageSource::Url(url)) => json!({
            "type": "image",
            "source": { "type": "url", "url": url },
        }),
    }
}

/// Hash of a system prompt plus turns. Text is trimmed and assistant turns
/// are reduced to their text, so a client that strips whitespace from a reply
/// or wraps it differently still lands on the same key.
fn key_of<'a>(system: &str, turns: impl Iterator<Item = &'a Turn>) -> String {
    let turns: Vec<Value> = turns.map(canonical_turn).collect();
    // serde_json maps are ordered, so the serialization is stable.
    let doc = json!({ "v": 1, "system": system.trim(), "turns": turns });
    hex_digest(doc.to_string().as_bytes())
}

fn canonical_turn(turn: &Turn) -> Value {
    match turn.role {
        Role::Assistant => {
            let text: String = turn
                .blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Text(s) => Some(s.as_str()),
                    Block::Image(_) => None,
                })
                .collect();
            json!({ "assistant": text.trim() })
        }
        Role::User => {
            let blocks: Vec<Value> = turn
                .blocks
                .iter()
                .map(|b| match b {
                    Block::Text(s) => json!({ "text": s.trim() }),
                    Block::Image(ImageSource::Base64 { data, .. }) => {
                        json!({ "image": hex_digest(data.as_bytes()) })
                    }
                    Block::Image(ImageSource::Url(url)) => json!({ "image_url": url }),
                })
                .collect();
            json!({ "user": blocks })
        }
    }
}

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> Block {
        Block::Text(s.to_string())
    }

    fn conversation(system: &str, turns: &[(Role, &str)]) -> Conversation {
        let mut b = ConversationBuilder::new();
        b.system(system);
        for (role, t) in turns {
            b.push(*role, vec![text(t)]);
        }
        b.build().unwrap()
    }

    fn parse_line(line: &str) -> Value {
        assert!(line.ends_with('\n'));
        serde_json::from_str(line.trim_end()).unwrap()
    }

    #[test]
    fn builder_merges_consecutive_same_role_messages() {
        let mut b = ConversationBuilder::new();
        b.push(Role::User, vec![text("a")]);
        b.push(Role::User, vec![text("b")]);
        let c = b.build().unwrap();
        assert_eq!(c.turns().len(), 1);
        assert_eq!(c.last().blocks, vec![text("a"), text("b")]);
    }

    #[test]
    fn builder_drops_blank_messages() {
        let mut b = ConversationBuilder::new();
        b.push(Role::User, vec![text("hi")]);
        b.push(Role::Assistant, vec![text("   ")]);
        b.push(Role::User, vec![text("again")]);
        let c = b.build().unwrap();
        assert_eq!(c.turns().len(), 1, "the blank assistant turn disappears and the users merge");
    }

    #[test]
    fn builder_rejects_empty_and_assistant_last() {
        assert!(ConversationBuilder::new().build().is_err());
        let mut b = ConversationBuilder::new();
        b.push(Role::User, vec![text("hi")]);
        b.push(Role::Assistant, vec![text("hello")]);
        assert_eq!(b.build().unwrap_err(), "the last message must come from the user");
    }

    #[test]
    fn builder_joins_system_parts() {
        let mut b = ConversationBuilder::new();
        b.system("one");
        b.system("  ");
        b.system("two");
        b.push(Role::User, vec![text("hi")]);
        assert_eq!(b.build().unwrap().system, "one\n\ntwo");
    }

    #[test]
    fn first_message_has_no_history_key() {
        assert_eq!(conversation("", &[(Role::User, "hi")]).history_key(), None);
    }

    #[test]
    fn next_request_finds_the_key_stored_after_the_reply() {
        let first = conversation("sys", &[(Role::User, "hi")]);
        let stored = first.key_after_reply("Hello!");
        let second = conversation(
            "sys",
            &[(Role::User, "hi"), (Role::Assistant, "Hello!"), (Role::User, "how are you?")],
        );
        assert_eq!(second.history_key(), Some(stored));
    }

    #[test]
    fn key_ignores_surrounding_whitespace_of_the_reply() {
        let first = conversation("", &[(Role::User, "hi")]);
        let second = conversation(
            "",
            &[(Role::User, "hi"), (Role::Assistant, "\n Hello! \n"), (Role::User, "next")],
        );
        assert_eq!(second.history_key(), Some(first.key_after_reply("Hello!")));
    }

    #[test]
    fn key_depends_on_system_prompt_and_content() {
        let a = conversation("one", &[(Role::User, "hi")]).key_after_reply("x");
        let b = conversation("two", &[(Role::User, "hi")]).key_after_reply("x");
        let c = conversation("one", &[(Role::User, "hi")]).key_after_reply("y");
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn key_distinguishes_images() {
        let with_image = |data: &str| {
            let mut b = ConversationBuilder::new();
            b.push(
                Role::User,
                vec![
                    text("what is this?"),
                    Block::Image(ImageSource::Base64 {
                        media_type: "image/png".into(),
                        data: data.into(),
                    }),
                ],
            );
            b.build().unwrap().key_after_reply("a square")
        };
        assert_ne!(with_image("AAAA"), with_image("BBBB"));
    }

    #[test]
    fn continuation_input_sends_only_the_new_message() {
        let c = conversation("", &[(Role::User, "hi"), (Role::Assistant, "hello"), (Role::User, "next")]);
        let line = parse_line(&c.continuation_input());
        assert_eq!(line["type"], "user");
        assert_eq!(line["message"]["role"], "user");
        assert_eq!(line["message"]["content"], json!([{ "type": "text", "text": "next" }]));
    }

    #[test]
    fn fresh_input_without_history_is_the_message_itself() {
        let c = conversation("", &[(Role::User, "hi")]);
        assert_eq!(c.fresh_input(), c.continuation_input());
    }

    #[test]
    fn fresh_input_replays_history_with_images_in_place() {
        let image = Block::Image(ImageSource::Url("https://example.com/cat.png".into()));
        let mut b = ConversationBuilder::new();
        b.push(Role::User, vec![text("look"), image.clone()]);
        b.push(Role::Assistant, vec![text("a cat")]);
        b.push(Role::User, vec![text("what color?")]);
        let line = parse_line(&b.build().unwrap().fresh_input());
        let content = line["message"]["content"].as_array().unwrap();

        assert_eq!(content.len(), 3, "text, image, text");
        let before = content[0]["text"].as_str().unwrap();
        assert!(before.contains("<conversation_history>\n<user>\nlook"));
        assert_eq!(content[1]["source"]["url"], "https://example.com/cat.png");
        let after = content[2]["text"].as_str().unwrap();
        assert!(after.contains("<assistant>\na cat\n</assistant>"));
        assert!(after.ends_with("Reply to the latest user message:\n\nwhat color?"));
    }

    #[test]
    fn base64_image_block_shape() {
        let block = block_json(&Block::Image(ImageSource::Base64 {
            media_type: "image/jpeg".into(),
            data: "QUJD".into(),
        }));
        assert_eq!(
            block,
            json!({ "type": "image", "source": { "type": "base64", "media_type": "image/jpeg", "data": "QUJD" } })
        );
    }
}
