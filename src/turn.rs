//! One chat turn from start to finish: continue a saved session or start a
//! fresh one, run the CLI, relay what it says, and remember the session for
//! the next turn. Routes only format the resulting `TurnEvent`s.

use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::conversation::Conversation;
use crate::server::AppState;
use crate::session::SessionStore;
use crate::status::RuntimeStatus;
use crate::subprocess::{self, SubprocessEvent, SubprocessOptions};
use crate::types::claude_cli::{ResultMessage, ResultUsage};

#[derive(Debug)]
pub enum TurnEvent {
    /// The CLI started; `model` is the real model id.
    Started { model: String },
    Delta(String),
    Finished(TurnOutput),
    Failed(TurnError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnOutput {
    pub text: String,
    pub model: String,
    /// Messages API stop reason: `end_turn`, `max_tokens`, `refusal`, …
    pub stop_reason: String,
    pub usage: ResultUsage,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurnError {
    /// HTTP status to answer with.
    pub status: u16,
    pub message: String,
}

pub struct TurnRequest {
    pub request_id: String,
    pub api: &'static str,
    /// What goes to `--model`: an alias or a full id.
    pub model: String,
    pub conversation: Conversation,
}

/// Start the turn in the background and return its events.
pub fn start(state: &AppState, request: TurnRequest) -> mpsc::Receiver<TurnEvent> {
    let (tx, rx) = mpsc::channel(64);
    let state = state.clone();
    tokio::spawn(async move { drive(state, request, tx).await });
    rx
}

async fn drive(state: AppState, request: TurnRequest, tx: mpsc::Sender<TurnEvent>) {
    let rid = &request.request_id;
    let mut resume = match request.conversation.history_key() {
        Some(key) => state.sessions.lookup(&key).await,
        None => None,
    };
    if request.conversation.history_key().is_some() && resume.is_none() {
        info!("[req={rid}] History not seen before, replaying it into a fresh session");
    }

    loop {
        let input = match resume {
            Some(_) => request.conversation.continuation_input(),
            None => request.conversation.fresh_input(),
        };
        let options = SubprocessOptions {
            request_id: rid.clone(),
            model: request.model.clone(),
            system_prompt: request.conversation.system.clone(),
            resume: resume.clone(),
            cwd: state.cwd.clone(),
            api: request.api,
        };
        let (process_tx, process_rx) = mpsc::channel(64);
        tokio::spawn(subprocess::spawn_subprocess(input, options, process_tx));

        let resuming = resume.is_some();
        match relay(&state.sessions, &state.status, &request, resuming, process_rx, &tx).await {
            Relay::Done => return,
            Relay::ResumeFailed => {
                warn!("[req={rid}] Saved session could not be resumed, replaying the history instead");
                resume = None;
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum Relay {
    Done,
    /// The saved session is gone; nothing reached the client yet.
    ResumeFailed,
}

async fn relay(
    sessions: &SessionStore,
    status: &RuntimeStatus,
    request: &TurnRequest,
    resuming: bool,
    mut events: mpsc::Receiver<SubprocessEvent>,
    tx: &mpsc::Sender<TurnEvent>,
) -> Relay {
    let mut model = request.model.clone();
    let mut started = false;
    let mut streamed = String::new();
    let mut answered = false;
    let mut limit_rejected = false;

    while let Some(event) = events.recv().await {
        match event {
            SubprocessEvent::Init { session_id, model: resolved } => {
                info!("[req={}] Session {session_id} on {resolved}", request.request_id);
                if !resolved.is_empty() {
                    status.record_model(&request.model, &resolved);
                    model = resolved;
                }
                started = true;
                if tx.send(TurnEvent::Started { model: model.clone() }).await.is_err() {
                    return Relay::Done;
                }
            }
            SubprocessEvent::TextDelta(text) => {
                streamed.push_str(&text);
                if tx.send(TurnEvent::Delta(text)).await.is_err() {
                    return Relay::Done;
                }
            }
            SubprocessEvent::RateLimit(info) => {
                limit_rejected = info.status.as_deref().is_some_and(|s| s != "allowed");
                status.record_rate_limit(&info);
            }
            SubprocessEvent::Result(result) => {
                if let Some(usage) = &result.model_usage {
                    status.record_model_usage(usage);
                }
                if result.is_error {
                    if resuming && !started {
                        return Relay::ResumeFailed;
                    }
                    let _ = tx.send(TurnEvent::Failed(error_from_result(&result, limit_rejected))).await;
                } else {
                    let output = TurnOutput {
                        text: result.result.clone().unwrap_or_else(|| streamed.clone()),
                        model: model.clone(),
                        stop_reason: result.stop_reason.clone().unwrap_or_else(|| "end_turn".to_string()),
                        usage: result.usage.unwrap_or_default(),
                    };
                    if let Some(session_id) = result.session_id.as_ref().filter(|s| !s.is_empty()) {
                        remember(sessions, &request.conversation, &output.text, &streamed, session_id).await;
                    }
                    let _ = tx.send(TurnEvent::Finished(output)).await;
                }
                answered = true;
            }
            SubprocessEvent::Error(message) => {
                if !answered {
                    let _ = tx.send(TurnEvent::Failed(TurnError { status: 502, message })).await;
                    answered = true;
                }
            }
            SubprocessEvent::Timeout => {
                if !answered {
                    let error = TurnError {
                        status: 504,
                        message: "claude produced no output for 30 minutes".to_string(),
                    };
                    let _ = tx.send(TurnEvent::Failed(error)).await;
                    answered = true;
                }
            }
            SubprocessEvent::Close { code, stderr_tail } => {
                if !answered {
                    if resuming && !started {
                        return Relay::ResumeFailed;
                    }
                    let mut message = format!("claude exited with code {code} without a result");
                    if !stderr_tail.is_empty() {
                        message.push_str(": ");
                        message.push_str(&stderr_tail);
                    }
                    let _ = tx.send(TurnEvent::Failed(TurnError { status: 502, message })).await;
                }
                return Relay::Done;
            }
        }
    }
    Relay::Done
}

/// Remember the session under the key the client's next request will carry.
/// Streaming clients rebuild the reply from deltas, so when that text differs
/// from the final result it is registered too.
async fn remember(sessions: &SessionStore, conversation: &Conversation, text: &str, streamed: &str, session_id: &str) {
    sessions
        .remember(conversation.key_after_reply(text), session_id.to_string())
        .await;
    if !streamed.trim().is_empty() && streamed.trim() != text.trim() {
        sessions
            .remember(conversation.key_after_reply(streamed), session_id.to_string())
            .await;
    }
}

fn error_from_result(result: &ResultMessage, limit_rejected: bool) -> TurnError {
    let message = result
        .result
        .clone()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| {
            format!(
                "claude reported an error ({})",
                result.subtype.as_deref().unwrap_or("unknown")
            )
        });
    let status = match result.api_error_status {
        Some(status) if (400..600).contains(&status) => status,
        _ if limit_rejected => 429,
        _ => 502,
    };
    TurnError { status, message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{Block, ConversationBuilder, Role};
    use crate::types::claude_cli::RateLimitInfo;
    use std::collections::HashMap;

    fn request(turns: &[(Role, &str)]) -> TurnRequest {
        let mut b = ConversationBuilder::new();
        for (role, text) in turns {
            b.push(*role, vec![Block::Text(text.to_string())]);
        }
        TurnRequest {
            request_id: "r".into(),
            api: "openai",
            model: "haiku".into(),
            conversation: b.build().unwrap(),
        }
    }

    async fn sessions() -> SessionStore {
        let dir = std::env::temp_dir().join(format!("claude-max-api-turn-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        SessionStore::open(dir.join("sessions.json"), dir.join("t")).await
    }

    fn result(json: &str) -> SubprocessEvent {
        SubprocessEvent::Result(serde_json::from_str(json).unwrap())
    }

    fn init() -> SubprocessEvent {
        SubprocessEvent::Init {
            session_id: "s".into(),
            model: "claude-haiku-4-5-20251001".into(),
        }
    }

    fn close(code: i32) -> SubprocessEvent {
        SubprocessEvent::Close { code, stderr_tail: String::new() }
    }

    /// Feed `events` through `relay` and collect what the route would see.
    async fn run(
        req: &TurnRequest,
        store: &SessionStore,
        resuming: bool,
        events: Vec<SubprocessEvent>,
    ) -> (Relay, Vec<TurnEvent>) {
        let status = RuntimeStatus::new("test".into());
        let (ptx, prx) = mpsc::channel(64);
        for e in events {
            ptx.send(e).await.unwrap();
        }
        drop(ptx);
        let (tx, mut rx) = mpsc::channel(64);
        let outcome = relay(store, &status, req, resuming, prx, &tx).await;
        drop(tx);
        let mut out = Vec::new();
        while let Some(e) = rx.recv().await {
            out.push(e);
        }
        (outcome, out)
    }

    #[tokio::test]
    async fn success_streams_and_remembers_the_session() {
        let req = request(&[(Role::User, "hi")]);
        let store = sessions().await;
        let (outcome, events) = run(&req, &store, false, vec![
            init(),
            SubprocessEvent::TextDelta("Hel".into()),
            SubprocessEvent::TextDelta("lo".into()),
            result(r#"{"type":"result","subtype":"success","is_error":false,"result":"Hello","session_id":"sid-2","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2}}"#),
            close(0),
        ]).await;

        assert_eq!(outcome, Relay::Done);
        assert!(matches!(&events[0], TurnEvent::Started { model } if model == "claude-haiku-4-5-20251001"));
        assert!(matches!(&events[1], TurnEvent::Delta(t) if t == "Hel"));
        let TurnEvent::Finished(output) = &events[3] else { panic!("{events:?}") };
        assert_eq!(output.text, "Hello");
        assert_eq!(output.model, "claude-haiku-4-5-20251001");
        assert_eq!(output.usage.input_tokens, 10);
        assert_eq!(events.len(), 4);

        let next = request(&[(Role::User, "hi"), (Role::Assistant, "Hello"), (Role::User, "more")]);
        let key = next.conversation.history_key().unwrap();
        assert_eq!(store.lookup(&key).await.as_deref(), Some("sid-2"));
    }

    #[tokio::test]
    async fn missing_session_before_init_asks_for_a_retry() {
        let req = request(&[(Role::User, "hi"), (Role::Assistant, "Hello"), (Role::User, "more")]);
        let (outcome, events) = run(&req, &sessions().await, true, vec![
            result(r#"{"type":"result","subtype":"error_during_execution","is_error":true}"#),
            close(1),
        ]).await;
        assert_eq!(outcome, Relay::ResumeFailed);
        assert!(events.is_empty(), "nothing reaches the client before the retry");
    }

    #[tokio::test]
    async fn fresh_session_errors_are_reported_not_retried() {
        let req = request(&[(Role::User, "hi")]);
        let (outcome, events) = run(&req, &sessions().await, false, vec![
            result(r#"{"type":"result","subtype":"error_during_execution","is_error":true}"#),
            close(1),
        ]).await;
        assert_eq!(outcome, Relay::Done);
        assert!(matches!(&events[..], [TurnEvent::Failed(e)] if e.status == 502));
    }

    #[tokio::test]
    async fn api_errors_keep_their_status() {
        let req = request(&[(Role::User, "hi")]);
        let (_, events) = run(&req, &sessions().await, false, vec![
            init(),
            result(r#"{"type":"result","subtype":"success","is_error":true,"api_error_status":400,"result":"API Error: 400 Unable to download the file."}"#),
            close(1),
        ]).await;
        let TurnEvent::Failed(error) = &events[1] else { panic!("{events:?}") };
        assert_eq!(error.status, 400);
        assert!(error.message.contains("Unable to download"));
        assert_eq!(events.len(), 2, "the close after an answered error adds nothing");
    }

    #[tokio::test]
    async fn rejected_limit_turns_errors_into_429() {
        let req = request(&[(Role::User, "hi")]);
        let info = RateLimitInfo { status: Some("rejected".into()), windows: HashMap::new() };
        let (_, events) = run(&req, &sessions().await, false, vec![
            init(),
            SubprocessEvent::RateLimit(info),
            result(r#"{"type":"result","subtype":"success","is_error":true,"result":"You've hit your limit"}"#),
            close(1),
        ]).await;
        assert!(matches!(&events[1], TurnEvent::Failed(e) if e.status == 429));
    }

    #[tokio::test]
    async fn exit_without_result_reports_stderr() {
        let req = request(&[(Role::User, "hi")]);
        let (_, events) = run(&req, &sessions().await, false, vec![
            SubprocessEvent::Close { code: 1, stderr_tail: "Invalid API key · Please run /login".into() },
        ]).await;
        let [TurnEvent::Failed(error)] = &events[..] else { panic!("{events:?}") };
        assert_eq!(error.status, 502);
        assert!(error.message.ends_with("Please run /login"));
    }

    #[tokio::test]
    async fn timeout_is_a_504() {
        let req = request(&[(Role::User, "hi")]);
        let (_, events) = run(&req, &sessions().await, false, vec![init(), SubprocessEvent::Timeout]).await;
        assert!(matches!(&events[1], TurnEvent::Failed(e) if e.status == 504));
    }

    #[tokio::test]
    async fn streamed_text_is_remembered_when_it_differs_from_the_result() {
        let req = request(&[(Role::User, "hi")]);
        let store = sessions().await;
        run(&req, &store, false, vec![
            init(),
            SubprocessEvent::TextDelta("Part one. ".into()),
            SubprocessEvent::TextDelta("Part two.".into()),
            result(r#"{"type":"result","subtype":"success","is_error":false,"result":"Part two.","session_id":"sid-3"}"#),
            close(0),
        ]).await;
        let next = request(&[(Role::User, "hi"), (Role::Assistant, "Part one. Part two."), (Role::User, "and?")]);
        assert_eq!(store.lookup(&next.conversation.history_key().unwrap()).await.as_deref(), Some("sid-3"));
    }
}
