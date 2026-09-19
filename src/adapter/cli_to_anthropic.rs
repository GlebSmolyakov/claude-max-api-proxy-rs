//! Turn results → Anthropic Messages responses and stream events.

use serde_json::{Value, json};

use crate::error::AppError;
use crate::turn::{TurnEvent, TurnOutput};
use crate::types::anthropic::{MessagesResponse, ResponseBlock, Usage};
use crate::types::claude_cli::ResultUsage;

pub fn message(output: &TurnOutput, request_id: &str) -> MessagesResponse {
    let mut content = Vec::new();
    if !output.text.is_empty() || output.tool_calls.is_empty() {
        content.push(ResponseBlock::Text { text: output.text.clone() });
    }
    content.extend(output.tool_calls.iter().map(|call| ResponseBlock::ToolUse {
        id: call.id.clone(),
        name: call.name.clone(),
        input: call.input.clone(),
    }));
    MessagesResponse {
        id: format!("msg_{request_id}"),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: output.model.clone(),
        stop_reason: output.stop_reason.clone(),
        stop_sequence: None,
        usage: usage(&output.usage),
    }
}

pub fn usage(u: &ResultUsage) -> Usage {
    Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
        cache_creation_input_tokens: u.cache_creation_input_tokens,
        cache_read_input_tokens: u.cache_read_input_tokens,
    }
}

/// Turns `TurnEvent`s into named SSE events of an Anthropic stream:
/// `message_start`, `ping`, then a text block and tool_use blocks, each
/// opened, filled and stopped, then `message_delta` and `message_stop`.
pub struct AnthropicStream {
    id: String,
    model: String,
    started: bool,
    /// Index of the open text block, if one is open.
    text_block: Option<u32>,
    next_index: u32,
    closed: bool,
}

pub type SseEvent = (&'static str, String);

impl AnthropicStream {
    pub fn new(request_id: &str, model: &str) -> Self {
        Self {
            id: format!("msg_{request_id}"),
            model: model.to_string(),
            started: false,
            text_block: None,
            next_index: 0,
            closed: false,
        }
    }

    pub fn on_event(&mut self, event: TurnEvent) -> Vec<SseEvent> {
        if self.closed {
            return vec![];
        }
        match event {
            TurnEvent::Started { model } => {
                self.model = model;
                vec![]
            }
            TurnEvent::Delta(text) => {
                let mut out = self.start();
                out.extend(self.text(text));
                out
            }
            TurnEvent::Finished(output) => {
                if !self.started {
                    self.model = output.model.clone();
                }
                let mut out = self.start();
                let no_text_yet = self.next_index == 0;
                if no_text_yet && (!output.text.is_empty() || output.tool_calls.is_empty()) {
                    out.extend(self.text(output.text.clone()));
                }
                out.extend(self.close_text());
                for call in &output.tool_calls {
                    let index = self.next_index;
                    self.next_index += 1;
                    out.push(event_of(
                        "content_block_start",
                        json!({
                            "type": "content_block_start", "index": index,
                            "content_block": { "type": "tool_use", "id": call.id, "name": call.name, "input": {} },
                        }),
                    ));
                    out.push(event_of(
                        "content_block_delta",
                        json!({
                            "type": "content_block_delta", "index": index,
                            "delta": { "type": "input_json_delta", "partial_json": call.input.to_string() },
                        }),
                    ));
                    out.push(event_of("content_block_stop", json!({ "type": "content_block_stop", "index": index })));
                }
                out.push(event_of(
                    "message_delta",
                    json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": output.stop_reason, "stop_sequence": null },
                        "usage": usage(&output.usage),
                    }),
                ));
                out.push(event_of("message_stop", json!({ "type": "message_stop" })));
                self.closed = true;
                out
            }
            TurnEvent::Failed(error) => {
                self.closed = true;
                vec![event_of("error", AppError::upstream(error).anthropic_body())]
            }
        }
    }

    /// Called when the events run out; closes a stream that never finished.
    pub fn on_end(&mut self) -> Vec<SseEvent> {
        if self.closed {
            return vec![];
        }
        self.closed = true;
        let error = AppError::Internal("the turn ended without a result".to_string());
        vec![event_of("error", error.anthropic_body())]
    }

    /// `message_start` and `ping`, once.
    fn start(&mut self) -> Vec<SseEvent> {
        if self.started {
            return vec![];
        }
        self.started = true;
        vec![
            event_of(
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": self.id, "type": "message", "role": "assistant", "content": [],
                        "model": self.model, "stop_reason": null, "stop_sequence": null,
                        // Real counts arrive with message_delta.
                        "usage": Usage::default(),
                    },
                }),
            ),
            event_of("ping", json!({ "type": "ping" })),
        ]
    }

    /// A text delta, opening the text block first if needed.
    fn text(&mut self, text: String) -> Vec<SseEvent> {
        let mut out = Vec::new();
        let index = match self.text_block {
            Some(index) => index,
            None => {
                let index = self.next_index;
                self.next_index += 1;
                self.text_block = Some(index);
                out.push(event_of(
                    "content_block_start",
                    json!({ "type": "content_block_start", "index": index, "content_block": { "type": "text", "text": "" } }),
                ));
                index
            }
        };
        out.push(event_of(
            "content_block_delta",
            json!({ "type": "content_block_delta", "index": index, "delta": { "type": "text_delta", "text": text } }),
        ));
        out
    }

    fn close_text(&mut self) -> Vec<SseEvent> {
        match self.text_block.take() {
            Some(index) => vec![event_of("content_block_stop", json!({ "type": "content_block_stop", "index": index }))],
            None => vec![],
        }
    }
}

fn event_of(name: &'static str, data: Value) -> SseEvent {
    (name, data.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::ToolCall;
    use crate::turn::TurnError;

    fn output(text: &str) -> TurnOutput {
        TurnOutput {
            text: text.into(),
            model: "claude-haiku-4-5-20251001".into(),
            stop_reason: "end_turn".into(),
            usage: ResultUsage { input_tokens: 10, cache_creation_input_tokens: 0, cache_read_input_tokens: 3, output_tokens: 7 },
            tool_calls: vec![],
        }
    }

    fn tool_step(text: &str) -> TurnOutput {
        TurnOutput {
            stop_reason: "tool_use".into(),
            tool_calls: vec![ToolCall { id: "toolu_1".into(), name: "read_file".into(), input: json!({ "path": "a.rs" }) }],
            ..output(text)
        }
    }

    fn names(events: &[SseEvent]) -> Vec<&str> {
        events.iter().map(|(name, _)| *name).collect()
    }

    fn data(event: &SseEvent) -> Value {
        serde_json::from_str(&event.1).unwrap()
    }

    #[test]
    fn message_carries_model_stop_reason_and_usage() {
        let mut out = output("Hi");
        out.stop_reason = "max_tokens".into();
        let v = serde_json::to_value(message(&out, "abc")).unwrap();
        assert_eq!(v["id"], "msg_abc");
        assert_eq!(v["model"], "claude-haiku-4-5-20251001");
        assert_eq!(v["stop_reason"], "max_tokens");
        assert_eq!(v["content"], json!([{ "type": "text", "text": "Hi" }]));
        assert_eq!(v["usage"]["input_tokens"], 10);
        assert_eq!(v["usage"]["cache_read_input_tokens"], 3);
    }

    #[test]
    fn message_with_tool_use() {
        let v = serde_json::to_value(message(&tool_step(""), "abc")).unwrap();
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(
            v["content"],
            json!([{ "type": "tool_use", "id": "toolu_1", "name": "read_file", "input": { "path": "a.rs" } }]),
            "no empty text block before the call"
        );
        let with_text = serde_json::to_value(message(&tool_step("Let me look."), "abc")).unwrap();
        assert_eq!(with_text["content"][0]["type"], "text");
        assert_eq!(with_text["content"][1]["type"], "tool_use");
    }

    #[test]
    fn stream_sequence() {
        let mut s = AnthropicStream::new("abc", "haiku");
        assert!(s.on_event(TurnEvent::Started { model: "claude-haiku-4-5-20251001".into() }).is_empty());
        let first = s.on_event(TurnEvent::Delta("Hel".into()));
        assert_eq!(names(&first), ["message_start", "ping", "content_block_start", "content_block_delta"]);
        assert_eq!(data(&first[0])["message"]["model"], "claude-haiku-4-5-20251001");
        assert_eq!(data(&first[3])["delta"]["text"], "Hel");
        assert_eq!(names(&s.on_event(TurnEvent::Delta("lo".into()))), ["content_block_delta"]);

        let end = s.on_event(TurnEvent::Finished(output("Hello")));
        assert_eq!(names(&end), ["content_block_stop", "message_delta", "message_stop"]);
        assert_eq!(data(&end[1])["delta"]["stop_reason"], "end_turn");
        assert_eq!(data(&end[1])["usage"]["output_tokens"], 7);
        assert!(s.on_end().is_empty());
    }

    #[test]
    fn stream_with_text_then_tool_use() {
        let mut s = AnthropicStream::new("abc", "haiku");
        s.on_event(TurnEvent::Delta("Let me look.".into()));
        let end = s.on_event(TurnEvent::Finished(tool_step("Let me look.")));
        assert_eq!(
            names(&end),
            ["content_block_stop", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
        );
        let start = data(&end[1]);
        assert_eq!(start["index"], 1);
        assert_eq!(start["content_block"]["type"], "tool_use");
        assert_eq!(start["content_block"]["name"], "read_file");
        let delta = data(&end[2]);
        assert_eq!(delta["delta"]["type"], "input_json_delta");
        assert_eq!(delta["delta"]["partial_json"], r#"{"path":"a.rs"}"#);
        assert_eq!(data(&end[4])["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn stream_of_only_tool_use_has_no_text_block() {
        let mut s = AnthropicStream::new("abc", "haiku");
        let events = s.on_event(TurnEvent::Finished(tool_step("")));
        assert_eq!(
            names(&events),
            ["message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
        );
        assert_eq!(data(&events[2])["index"], 0);
        assert_eq!(data(&events[2])["content_block"]["type"], "tool_use");
    }

    #[test]
    fn stream_without_deltas_still_carries_the_text() {
        let mut s = AnthropicStream::new("abc", "haiku");
        let events = s.on_event(TurnEvent::Finished(output("Hi")));
        assert_eq!(
            names(&events),
            ["message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn stream_failure_is_an_error_event() {
        let mut s = AnthropicStream::new("abc", "haiku");
        s.on_event(TurnEvent::Delta("Hi".into()));
        let out = s.on_event(TurnEvent::Failed(TurnError { status: 529, message: "busy".into() }));
        assert_eq!(names(&out), ["error"]);
        assert_eq!(data(&out[0])["error"]["type"], "overloaded_error");
        assert!(s.on_event(TurnEvent::Delta("late".into())).is_empty());
    }

    #[test]
    fn unfinished_stream_is_closed_with_an_error() {
        let mut s = AnthropicStream::new("abc", "haiku");
        assert_eq!(names(&s.on_end()), ["error"]);
    }
}
