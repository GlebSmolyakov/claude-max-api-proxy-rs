//! Turn results → OpenAI chat completion responses and stream chunks.

use crate::error::AppError;
use crate::status::unix_now;
use crate::turn::{TurnEvent, TurnOutput};
use crate::types::claude_cli::ResultUsage;
use crate::types::openai::{
    ChatCompletionChunk, ChatCompletionResponse, Choice, ChunkChoice, ChunkDelta, PromptTokensDetails,
    ResponseMessage, Usage,
};

pub fn completion(output: &TurnOutput, request_id: &str) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: format!("chatcmpl-{request_id}"),
        object: "chat.completion".to_string(),
        created: unix_now(),
        model: output.model.clone(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: output.text.clone(),
            },
            finish_reason: finish_reason(&output.stop_reason).to_string(),
        }],
        usage: usage(&output.usage),
    }
}

/// Messages API stop reason → OpenAI finish reason.
pub fn finish_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "max_tokens" | "model_context_window_exceeded" => "length",
        "refusal" => "content_filter",
        _ => "stop",
    }
}

/// OpenAI counts cached input inside `prompt_tokens` and reports it again
/// under `prompt_tokens_details.cached_tokens`.
pub fn usage(u: &ResultUsage) -> Usage {
    let prompt = u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens;
    Usage {
        prompt_tokens: prompt,
        completion_tokens: u.output_tokens,
        total_tokens: prompt + u.output_tokens,
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: u.cache_read_input_tokens,
        },
    }
}

/// Turns `TurnEvent`s into the `data:` payloads of an OpenAI SSE stream.
pub struct OpenAiStream {
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    sent_content: bool,
    closed: bool,
}

impl OpenAiStream {
    pub fn new(request_id: &str, model: &str, include_usage: bool) -> Self {
        Self {
            id: format!("chatcmpl-{request_id}"),
            created: unix_now(),
            model: model.to_string(),
            include_usage,
            sent_content: false,
            closed: false,
        }
    }

    pub fn on_event(&mut self, event: TurnEvent) -> Vec<String> {
        if self.closed {
            return vec![];
        }
        match event {
            TurnEvent::Started { model } => {
                self.model = model;
                vec![]
            }
            TurnEvent::Delta(text) => vec![self.content_chunk(text)],
            TurnEvent::Finished(output) => {
                let mut out = Vec::new();
                self.model = output.model.clone();
                if !self.sent_content && !output.text.is_empty() {
                    out.push(self.content_chunk(output.text.clone()));
                }
                let role = (!self.sent_content).then(|| "assistant".to_string());
                out.push(self.chunk(
                    vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta { role, content: None },
                        finish_reason: Some(finish_reason(&output.stop_reason).to_string()),
                    }],
                    None,
                ));
                if self.include_usage {
                    out.push(self.chunk(vec![], Some(usage(&output.usage))));
                }
                out.push("[DONE]".to_string());
                self.closed = true;
                out
            }
            TurnEvent::Failed(error) => {
                self.closed = true;
                vec![AppError::upstream(error).openai_body().to_string(), "[DONE]".to_string()]
            }
        }
    }

    /// Called when the events run out; closes a stream that never finished.
    pub fn on_end(&mut self) -> Vec<String> {
        if self.closed {
            return vec![];
        }
        self.closed = true;
        let error = AppError::Internal("the turn ended without a result".to_string());
        vec![error.openai_body().to_string(), "[DONE]".to_string()]
    }

    fn content_chunk(&mut self, text: String) -> String {
        let role = (!self.sent_content).then(|| "assistant".to_string());
        self.sent_content = true;
        self.chunk(
            vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta { role, content: Some(text) },
                finish_reason: None,
            }],
            None,
        )
    }

    fn chunk(&self, choices: Vec<ChunkChoice>, usage: Option<Usage>) -> String {
        let chunk = ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices,
            usage,
        };
        serde_json::to_string(&chunk).expect("chunks serialize")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn::TurnError;
    use serde_json::Value;

    fn output(text: &str) -> TurnOutput {
        TurnOutput {
            text: text.into(),
            model: "claude-haiku-4-5-20251001".into(),
            stop_reason: "end_turn".into(),
            usage: ResultUsage {
                input_tokens: 10,
                cache_creation_input_tokens: 5,
                cache_read_input_tokens: 100,
                output_tokens: 7,
            },
        }
    }

    fn parse(payload: &str) -> Value {
        serde_json::from_str(payload).unwrap()
    }

    #[test]
    fn completion_reports_real_model_and_usage() {
        let r = serde_json::to_value(completion(&output("Hi"), "abc")).unwrap();
        assert_eq!(r["id"], "chatcmpl-abc");
        assert_eq!(r["model"], "claude-haiku-4-5-20251001");
        assert_eq!(r["choices"][0]["message"]["content"], "Hi");
        assert_eq!(r["choices"][0]["finish_reason"], "stop");
        assert_eq!(r["usage"]["prompt_tokens"], 115);
        assert_eq!(r["usage"]["completion_tokens"], 7);
        assert_eq!(r["usage"]["total_tokens"], 122);
        assert_eq!(r["usage"]["prompt_tokens_details"]["cached_tokens"], 100);
    }

    #[test]
    fn finish_reasons() {
        assert_eq!(finish_reason("end_turn"), "stop");
        assert_eq!(finish_reason("stop_sequence"), "stop");
        assert_eq!(finish_reason("max_tokens"), "length");
        assert_eq!(finish_reason("refusal"), "content_filter");
    }

    #[test]
    fn stream_sequence() {
        let mut s = OpenAiStream::new("abc", "haiku", false);
        assert!(s.on_event(TurnEvent::Started { model: "claude-haiku-4-5-20251001".into() }).is_empty());
        let first = parse(&s.on_event(TurnEvent::Delta("Hel".into()))[0]);
        assert_eq!(first["model"], "claude-haiku-4-5-20251001");
        assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(first["choices"][0]["delta"]["content"], "Hel");
        let second = parse(&s.on_event(TurnEvent::Delta("lo".into()))[0]);
        assert!(second["choices"][0]["delta"].get("role").is_none());

        let end = s.on_event(TurnEvent::Finished(output("Hello")));
        assert_eq!(end.len(), 2, "finish chunk and [DONE]; no usage chunk unless asked");
        assert_eq!(parse(&end[0])["choices"][0]["finish_reason"], "stop");
        assert_eq!(end[1], "[DONE]");
        assert!(s.on_event(TurnEvent::Delta("late".into())).is_empty());
        assert!(s.on_end().is_empty());
    }

    #[test]
    fn stream_usage_chunk_when_requested() {
        let mut s = OpenAiStream::new("abc", "haiku", true);
        s.on_event(TurnEvent::Delta("Hi".into()));
        let end = s.on_event(TurnEvent::Finished(output("Hi")));
        let usage = parse(&end[1]);
        assert_eq!(usage["choices"], Value::Array(vec![]));
        assert_eq!(usage["usage"]["completion_tokens"], 7);
        assert_eq!(end[2], "[DONE]");
    }

    #[test]
    fn stream_without_deltas_still_carries_the_text() {
        let mut s = OpenAiStream::new("abc", "haiku", false);
        let end = s.on_event(TurnEvent::Finished(output("Hi")));
        assert_eq!(parse(&end[0])["choices"][0]["delta"]["content"], "Hi");
        assert_eq!(end.len(), 3);
    }

    #[test]
    fn stream_failure_sends_an_error_and_done() {
        let mut s = OpenAiStream::new("abc", "haiku", false);
        let out = s.on_event(TurnEvent::Failed(TurnError { status: 429, message: "limit".into() }));
        assert_eq!(parse(&out[0])["error"]["type"], "rate_limit_error");
        assert_eq!(out[1], "[DONE]");
    }

    #[test]
    fn unfinished_stream_is_closed_with_an_error() {
        let mut s = OpenAiStream::new("abc", "haiku", false);
        s.on_event(TurnEvent::Delta("Hi".into()));
        let out = s.on_end();
        assert_eq!(parse(&out[0])["error"]["type"], "server_error");
        assert_eq!(out[1], "[DONE]");
    }
}
