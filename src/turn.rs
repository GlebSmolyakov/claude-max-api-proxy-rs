//! One chat turn from start to finish: continue a saved session or start a
//! fresh one, run the CLI, relay what it says, and remember the session for
//! the next turn. Routes only format the resulting `TurnEvent`s.
//!
//! A turn with client tools can span several HTTP requests. When the model
//! calls a tool, the turn answers the current request with the call and
//! parks: the CLI process stays alive, waiting in the MCP bridge. The
//! client's next request carries the result, finds the parked turn by the
//! tool call id, and the same process goes on.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::bridge::{Bridge, ToolOutcome};
use crate::conversation::{Block, Conversation, ToolCall};
use crate::server::AppState;
use crate::session::SessionStore;
use crate::status::RuntimeStatus;
use crate::subprocess::{self, INACTIVITY_TIMEOUT, Process, SubprocessEvent, SubprocessOptions};
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
    /// Messages API stop reason: `end_turn`, `tool_use`, `max_tokens`, …
    pub stop_reason: String,
    /// Tokens of this step: one API call.
    pub usage: ResultUsage,
    /// Tools the client should run, when `stop_reason` is `tool_use`.
    pub tool_calls: Vec<ToolCall>,
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

    let results = request.conversation.tool_results();
    if let Some(first) = results.first() {
        if let Some(parked) = state.pending.take(&first.tool_use_id) {
            continue_parked(&state, &request, parked, &tx).await;
            return;
        }
        info!("[req={rid}] Tool results for a turn this proxy no longer holds, replaying the history");
    }

    let mut resume = match request.conversation.history_key() {
        Some(key) => state.sessions.lookup(&key).await,
        None => None,
    };
    if request.conversation.history_key().is_some() && resume.is_none() && results.is_empty() {
        info!("[req={rid}] History not seen before, replaying it into a fresh session");
    }

    loop {
        let bridge = (!request.conversation.tools.is_empty())
            .then(|| state.bridges.register(request.conversation.tools.clone()));
        let input = match resume {
            Some(_) => request.conversation.continuation_input(),
            None => request.conversation.fresh_input(),
        };
        let options = SubprocessOptions {
            request_id: rid.clone(),
            model: request.model.clone(),
            system_prompt: request.conversation.system.clone(),
            resume: resume.clone(),
            mcp_url: bridge.as_ref().map(|(token, _)| format!("{}/{token}", state.mcp_base)),
            cwd: state.cwd.clone(),
            api: request.api,
        };
        let process = subprocess::spawn(input, options);

        let ctx = RelayCtx {
            sessions: &state.sessions,
            status: &state.status,
            request: &request,
            bridge: bridge.as_ref().map(|(_, b)| b.as_ref()),
            resuming: resume.is_some(),
            started: false,
            model: request.model.clone(),
        };
        let outcome = relay(ctx, process, &tx).await;
        match (outcome, bridge) {
            (Relay::Parked { process, tool_ids, model }, Some((token, bridge))) => {
                info!("[req={rid}] Waiting for the client to run {} tool call(s)", tool_ids.len());
                state.pending.park(Parked { process, token, bridge, model, tool_ids, since: Instant::now() });
                return;
            }
            (Relay::ResumeFailed, bridge) => {
                if let Some((token, _)) = bridge {
                    state.bridges.remove(&token);
                }
                warn!("[req={rid}] Saved session could not be resumed, replaying the history instead");
                resume = None;
            }
            (_, bridge) => {
                if let Some((token, _)) = bridge {
                    state.bridges.remove(&token);
                }
                return;
            }
        }
    }
}

/// Hand the client's tool results to the parked process and relay what the
/// model does next.
async fn continue_parked(state: &AppState, request: &TurnRequest, parked: Parked, tx: &mpsc::Sender<TurnEvent>) {
    let rid = &request.request_id;
    let outcomes = outcomes_for(&request.conversation, &parked.tool_ids);
    let missing = outcomes.iter().filter(|(_, o)| o.is_none()).count();
    info!(
        "[req={rid}] Continuing a parked turn with {} of {} tool result(s)",
        outcomes.len() - missing,
        outcomes.len()
    );
    for (id, outcome) in outcomes {
        let outcome = outcome.unwrap_or_else(|| {
            warn!("[req={rid}] The client sent no result for tool call {id}");
            ToolOutcome {
                content: vec![Block::Text("The client returned no result for this call.".to_string())],
                is_error: true,
            }
        });
        parked.bridge.deliver(&id, outcome);
    }

    let Parked { process, token, bridge, model, .. } = parked;
    if tx.send(TurnEvent::Started { model: model.clone() }).await.is_err() {
        state.bridges.remove(&token);
        return;
    }
    let ctx = RelayCtx {
        sessions: &state.sessions,
        status: &state.status,
        request,
        bridge: Some(&bridge),
        resuming: false,
        started: true,
        model,
    };
    match relay(ctx, process, tx).await {
        Relay::Parked { process, tool_ids, model } => {
            info!("[req={rid}] Waiting for the client to run {} tool call(s)", tool_ids.len());
            state.pending.park(Parked { process, token, bridge, model, tool_ids, since: Instant::now() });
        }
        _ => state.bridges.remove(&token),
    }
}

/// The result for each of `tool_ids`, looked up across the whole
/// conversation; `None` where the client sent none. Text the user added next
/// to the results cannot become a new message in the middle of a turn, so it
/// rides along with the last result.
fn outcomes_for(conversation: &Conversation, tool_ids: &[String]) -> Vec<(String, Option<ToolOutcome>)> {
    let note = conversation.last_text();
    let mut outcomes: Vec<(String, Option<ToolOutcome>)> = tool_ids
        .iter()
        .map(|id| {
            let outcome = conversation
                .find_tool_result(id)
                .map(|r| ToolOutcome { content: r.content, is_error: r.is_error });
            (id.clone(), outcome)
        })
        .collect();
    if !note.is_empty()
        && let Some((_, Some(last))) = outcomes.iter_mut().rev().find(|(_, o)| o.is_some())
    {
        last.content.push(Block::Text(format!("\n\n[The user also wrote]\n{note}")));
    }
    outcomes
}

struct RelayCtx<'a> {
    sessions: &'a SessionStore,
    status: &'a RuntimeStatus,
    request: &'a TurnRequest,
    /// Present when the client declared tools.
    bridge: Option<&'a Bridge>,
    resuming: bool,
    /// `Started` was already sent (a continued turn).
    started: bool,
    model: String,
}

enum Relay {
    Done,
    /// The saved session is gone; nothing reached the client yet.
    ResumeFailed,
    /// The model called client tools; the process waits for their results.
    Parked { process: Process, tool_ids: Vec<String>, model: String },
}

async fn relay(ctx: RelayCtx<'_>, mut process: Process, tx: &mpsc::Sender<TurnEvent>) -> Relay {
    let mut model = ctx.model;
    let mut started = ctx.started;
    let mut streamed = String::new();
    let mut answered = false;
    let mut limit_rejected = false;
    let mut step_usage: Option<ResultUsage> = None;
    // Tool calls of the current step: announced ids, and complete calls.
    let mut announced: Vec<String> = Vec::new();
    let mut calls: Vec<ToolCall> = Vec::new();
    let mut tool_step_ended = false;

    while let Some(event) = process.events.recv().await {
        match event {
            SubprocessEvent::Init { session_id, model: resolved } => {
                info!("[req={}] Session {session_id} on {resolved}", ctx.request.request_id);
                if !resolved.is_empty() {
                    ctx.status.record_model(&ctx.request.model, &resolved);
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
            SubprocessEvent::ToolUseStarted(id) => announced.push(id),
            SubprocessEvent::ToolUse(mut call) => {
                if let Some(bridge) = ctx.bridge {
                    call.name = bridge.client_name(&call.name);
                }
                calls.push(call);
            }
            SubprocessEvent::StepEnd { stop_reason, usage } => {
                if usage.is_some() {
                    step_usage = usage;
                }
                tool_step_ended = ctx.bridge.is_some() && stop_reason.as_deref() == Some("tool_use");
            }
            SubprocessEvent::RateLimit(info) => {
                limit_rejected = info.status.as_deref().is_some_and(|s| s != "allowed");
                ctx.status.record_rate_limit(&info);
            }
            SubprocessEvent::Result(result) => {
                if let Some(usage) = &result.model_usage {
                    ctx.status.record_model_usage(usage);
                }
                if result.is_error {
                    if ctx.resuming && !started {
                        return Relay::ResumeFailed;
                    }
                    let error = error_from_result(&result, limit_rejected);
                    log_failure(&ctx.request.request_id, &error);
                    let _ = tx.send(TurnEvent::Failed(error)).await;
                } else {
                    let output = TurnOutput {
                        text: result.result.clone().unwrap_or_else(|| streamed.clone()),
                        model: model.clone(),
                        stop_reason: result.stop_reason.clone().unwrap_or_else(|| "end_turn".to_string()),
                        usage: step_usage.or(result.usage).unwrap_or_default(),
                        tool_calls: Vec::new(),
                    };
                    if let Some(session_id) = result.session_id.as_ref().filter(|s| !s.is_empty()) {
                        remember(ctx.sessions, &ctx.request.conversation, &output.text, &streamed, session_id).await;
                    }
                    let _ = tx.send(TurnEvent::Finished(output)).await;
                }
                answered = true;
            }
            SubprocessEvent::Error(message) => {
                if !answered {
                    let error = TurnError { status: 502, message };
                    log_failure(&ctx.request.request_id, &error);
                    let _ = tx.send(TurnEvent::Failed(error)).await;
                    answered = true;
                }
            }
            SubprocessEvent::Timeout => {
                if !answered {
                    let error = TurnError {
                        status: 504,
                        message: "claude produced no output for 30 minutes".to_string(),
                    };
                    log_failure(&ctx.request.request_id, &error);
                    let _ = tx.send(TurnEvent::Failed(error)).await;
                    answered = true;
                }
            }
            SubprocessEvent::Close { code, stderr_tail } => {
                if !answered {
                    if ctx.resuming && !started {
                        return Relay::ResumeFailed;
                    }
                    let mut message = format!("claude exited with code {code} without a result");
                    if !stderr_tail.is_empty() {
                        message.push_str(": ");
                        message.push_str(&stderr_tail);
                    }
                    let error = TurnError { status: 502, message };
                    log_failure(&ctx.request.request_id, &error);
                    let _ = tx.send(TurnEvent::Failed(error)).await;
                }
                return Relay::Done;
            }
        }

        // A tool step is complete once the step has ended and every
        // announced call has arrived with its input.
        let complete = tool_step_ended
            && !calls.is_empty()
            && announced.iter().all(|id| calls.iter().any(|c| &c.id == id));
        if complete {
            let output = TurnOutput {
                text: std::mem::take(&mut streamed),
                model: model.clone(),
                stop_reason: "tool_use".to_string(),
                usage: step_usage.unwrap_or_default(),
                tool_calls: calls.clone(),
            };
            if tx.send(TurnEvent::Finished(output)).await.is_err() {
                return Relay::Done;
            }
            let tool_ids = calls.into_iter().map(|c| c.id).collect();
            return Relay::Parked { process, tool_ids, model };
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

/// Every failure reaches the log with its message: a streaming client gets
/// it inside the stream, where it would otherwise be seen by no one else.
fn log_failure(request_id: &str, error: &TurnError) {
    warn!("[req={request_id}] Failed with {}: {}", error.status, error.message);
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

// ── Parked turns ────────────────────────────────────────────────

/// A turn waiting for the client to run tools. Dropping it kills the process.
pub struct Parked {
    process: Process,
    token: String,
    bridge: Arc<Bridge>,
    model: String,
    tool_ids: Vec<String>,
    since: Instant,
}

#[derive(Default)]
pub struct PendingTurns {
    inner: Mutex<PendingInner>,
}

#[derive(Default)]
struct PendingInner {
    by_tool: HashMap<String, u64>,
    turns: HashMap<u64, Parked>,
    next: u64,
}

impl PendingTurns {
    fn park(&self, parked: Parked) {
        let mut inner = self.inner.lock().unwrap();
        inner.next += 1;
        let key = inner.next;
        for id in &parked.tool_ids {
            inner.by_tool.insert(id.clone(), key);
        }
        inner.turns.insert(key, parked);
    }

    /// The turn waiting for `tool_use_id`, removed from the list.
    fn take(&self, tool_use_id: &str) -> Option<Parked> {
        let mut inner = self.inner.lock().unwrap();
        let key = inner.by_tool.get(tool_use_id).copied()?;
        let parked = inner.turns.remove(&key)?;
        for id in &parked.tool_ids {
            inner.by_tool.remove(id);
        }
        Some(parked)
    }

    fn expire(&self, max_age: Duration) -> Vec<Parked> {
        let mut inner = self.inner.lock().unwrap();
        let old: Vec<u64> = inner
            .turns
            .iter()
            .filter(|(_, p)| p.since.elapsed() >= max_age)
            .map(|(k, _)| *k)
            .collect();
        let mut expired = Vec::new();
        for key in old {
            if let Some(parked) = inner.turns.remove(&key) {
                for id in &parked.tool_ids {
                    inner.by_tool.remove(id);
                }
                expired.push(parked);
            }
        }
        expired
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().turns.len()
    }
}

/// Every minute, kill turns that waited for tool results longer than the
/// CLI would wait for them anyway.
pub fn spawn_expiry_task(state: AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        loop {
            interval.tick().await;
            for parked in state.pending.expire(INACTIVITY_TIMEOUT) {
                warn!("Dropping a turn that waited {:.0}s for tool results", parked.since.elapsed().as_secs_f64());
                state.bridges.remove(&parked.token);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{ConversationBuilder, Role, ToolDef};
    use crate::types::claude_cli::RateLimitInfo;
    use serde_json::json;

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
        SubprocessEvent::Init { session_id: "s".into(), model: "claude-haiku-4-5-20251001".into() }
    }

    fn close(code: i32) -> SubprocessEvent {
        SubprocessEvent::Close { code, stderr_tail: String::new() }
    }

    fn tool_use(id: &str) -> SubprocessEvent {
        SubprocessEvent::ToolUse(ToolCall { id: id.into(), name: "mcp__c__read_file".into(), input: json!({ "path": "a.rs" }) })
    }

    fn step_end(reason: &str) -> SubprocessEvent {
        SubprocessEvent::StepEnd {
            stop_reason: Some(reason.into()),
            usage: Some(ResultUsage { input_tokens: 100, output_tokens: 20, ..ResultUsage::default() }),
        }
    }

    fn tools_bridge() -> Bridge {
        Bridge::new(vec![ToolDef { name: "read_file".into(), description: String::new(), input_schema: json!({}) }])
    }

    /// Feed `events` through `relay` and collect what the route would see.
    async fn run(
        req: &TurnRequest,
        store: &SessionStore,
        bridge: Option<&Bridge>,
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
        let ctx = RelayCtx {
            sessions: store,
            status: &status,
            request: req,
            bridge,
            resuming,
            started: false,
            model: req.model.clone(),
        };
        let outcome = relay(ctx, Process::from_events(prx), &tx).await;
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
        let (outcome, events) = run(&req, &store, None, false, vec![
            init(),
            SubprocessEvent::TextDelta("Hel".into()),
            SubprocessEvent::TextDelta("lo".into()),
            step_end("end_turn"),
            result(r#"{"type":"result","subtype":"success","is_error":false,"result":"Hello","session_id":"sid-2","stop_reason":"end_turn","usage":{"input_tokens":999,"output_tokens":999}}"#),
            close(0),
        ]).await;

        assert!(matches!(outcome, Relay::Done));
        assert!(matches!(&events[0], TurnEvent::Started { model } if model == "claude-haiku-4-5-20251001"));
        assert!(matches!(&events[1], TurnEvent::Delta(t) if t == "Hel"));
        let TurnEvent::Finished(output) = &events[3] else { panic!("{events:?}") };
        assert_eq!(output.text, "Hello");
        assert_eq!(output.usage.input_tokens, 100, "the step's usage, not the run's total");
        assert!(output.tool_calls.is_empty());
        assert_eq!(events.len(), 4);

        let next = request(&[(Role::User, "hi"), (Role::Assistant, "Hello"), (Role::User, "more")]);
        let key = next.conversation.history_key().unwrap();
        assert_eq!(store.lookup(&key).await.as_deref(), Some("sid-2"));
    }

    #[tokio::test]
    async fn a_tool_step_answers_with_the_calls_and_parks() {
        let req = request(&[(Role::User, "what is in a.rs?")]);
        let bridge = tools_bridge();
        let (outcome, events) = run(&req, &sessions().await, Some(&bridge), false, vec![
            init(),
            SubprocessEvent::TextDelta("Let me look.".into()),
            SubprocessEvent::ToolUseStarted("toolu_1".into()),
            tool_use("toolu_1"),
            step_end("tool_use"),
        ]).await;

        let Relay::Parked { tool_ids, model, .. } = outcome else { panic!("not parked") };
        assert_eq!(tool_ids, ["toolu_1"]);
        assert_eq!(model, "claude-haiku-4-5-20251001");
        let TurnEvent::Finished(output) = events.last().unwrap() else { panic!("{events:?}") };
        assert_eq!(output.stop_reason, "tool_use");
        assert_eq!(output.text, "Let me look.");
        assert_eq!(output.tool_calls[0].name, "read_file", "the MCP prefix is gone");
        assert_eq!(output.tool_calls[0].input["path"], "a.rs");
        assert_eq!(output.usage.output_tokens, 20);
    }

    #[tokio::test]
    async fn a_step_waits_for_every_announced_call() {
        let req = request(&[(Role::User, "read both")]);
        let bridge = tools_bridge();
        let (outcome, events) = run(&req, &sessions().await, Some(&bridge), false, vec![
            init(),
            SubprocessEvent::ToolUseStarted("toolu_1".into()),
            SubprocessEvent::ToolUseStarted("toolu_2".into()),
            tool_use("toolu_1"),
            step_end("tool_use"),
            tool_use("toolu_2"),
        ]).await;
        let Relay::Parked { tool_ids, .. } = outcome else { panic!("not parked") };
        assert_eq!(tool_ids, ["toolu_1", "toolu_2"]);
        let TurnEvent::Finished(output) = events.last().unwrap() else { panic!() };
        assert_eq!(output.tool_calls.len(), 2);
    }

    #[tokio::test]
    async fn tool_calls_without_client_tools_do_not_park() {
        let req = request(&[(Role::User, "hi")]);
        let (outcome, _) = run(&req, &sessions().await, None, false, vec![
            init(),
            tool_use("toolu_1"),
            step_end("tool_use"),
            close(1),
        ]).await;
        assert!(matches!(outcome, Relay::Done));
    }

    #[tokio::test]
    async fn missing_session_before_init_asks_for_a_retry() {
        let req = request(&[(Role::User, "hi"), (Role::Assistant, "Hello"), (Role::User, "more")]);
        let (outcome, events) = run(&req, &sessions().await, None, true, vec![
            result(r#"{"type":"result","subtype":"error_during_execution","is_error":true}"#),
            close(1),
        ]).await;
        assert!(matches!(outcome, Relay::ResumeFailed));
        assert!(events.is_empty(), "nothing reaches the client before the retry");
    }

    #[tokio::test]
    async fn fresh_session_errors_are_reported_not_retried() {
        let req = request(&[(Role::User, "hi")]);
        let (outcome, events) = run(&req, &sessions().await, None, false, vec![
            result(r#"{"type":"result","subtype":"error_during_execution","is_error":true}"#),
            close(1),
        ]).await;
        assert!(matches!(outcome, Relay::Done));
        assert!(matches!(&events[..], [TurnEvent::Failed(e)] if e.status == 502));
    }

    #[tokio::test]
    async fn api_errors_keep_their_status() {
        let req = request(&[(Role::User, "hi")]);
        let (_, events) = run(&req, &sessions().await, None, false, vec![
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
        let (_, events) = run(&req, &sessions().await, None, false, vec![
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
        let (_, events) = run(&req, &sessions().await, None, false, vec![
            SubprocessEvent::Close { code: 1, stderr_tail: "Invalid API key · Please run /login".into() },
        ]).await;
        let [TurnEvent::Failed(error)] = &events[..] else { panic!("{events:?}") };
        assert_eq!(error.status, 502);
        assert!(error.message.ends_with("Please run /login"));
    }

    #[tokio::test]
    async fn timeout_is_a_504() {
        let req = request(&[(Role::User, "hi")]);
        let (_, events) = run(&req, &sessions().await, None, false, vec![init(), SubprocessEvent::Timeout]).await;
        assert!(matches!(&events[1], TurnEvent::Failed(e) if e.status == 504));
    }

    #[tokio::test]
    async fn streamed_text_is_remembered_when_it_differs_from_the_result() {
        let req = request(&[(Role::User, "hi")]);
        let store = sessions().await;
        run(&req, &store, None, false, vec![
            init(),
            SubprocessEvent::TextDelta("Part one. ".into()),
            SubprocessEvent::TextDelta("Part two.".into()),
            result(r#"{"type":"result","subtype":"success","is_error":false,"result":"Part two.","session_id":"sid-3"}"#),
            close(0),
        ]).await;
        let next = request(&[(Role::User, "hi"), (Role::Assistant, "Part one. Part two."), (Role::User, "and?")]);
        assert_eq!(store.lookup(&next.conversation.history_key().unwrap()).await.as_deref(), Some("sid-3"));
    }

    #[test]
    fn outcomes_come_from_anywhere_in_the_history() {
        let mut b = ConversationBuilder::new();
        let call = |id: &str| Block::ToolUse(ToolCall { id: id.into(), name: "shell".into(), input: json!({}) });
        let result = |id: &str, out: &str| Block::ToolResult { tool_use_id: id.into(), content: vec![Block::Text(out.into())], is_error: false };
        b.push(Role::User, vec![Block::Text("look".into())]);
        b.push(Role::Assistant, vec![call("t1")]);
        b.push(Role::User, vec![result("t1", "one")]);
        b.push(Role::Assistant, vec![call("t2")]);
        b.push(Role::User, vec![result("t2", "two"), Block::Text("and hurry".into())]);
        let c = b.build().unwrap();

        let outcomes = outcomes_for(&c, &["t1".into(), "t2".into(), "t3".into()]);
        assert_eq!(outcomes[0].1.as_ref().unwrap().content, vec![Block::Text("one".into())]);
        let second = &outcomes[1].1.as_ref().unwrap().content;
        assert_eq!(second[0], Block::Text("two".into()));
        assert!(matches!(&second[1], Block::Text(t) if t.ends_with("and hurry")), "the note rides with the last result");
        assert!(outcomes[2].1.is_none(), "t3 was never answered");
    }

    fn parked(ids: &[&str], age: Duration) -> Parked {
        let (_tx, rx) = mpsc::channel(1);
        Parked {
            process: Process::from_events(rx),
            token: "tok".into(),
            bridge: Arc::new(tools_bridge()),
            model: "m".into(),
            tool_ids: ids.iter().map(|s| s.to_string()).collect(),
            since: Instant::now() - age,
        }
    }

    #[test]
    fn a_parked_turn_is_found_by_any_of_its_calls_once() {
        let pending = PendingTurns::default();
        pending.park(parked(&["t1", "t2"], Duration::ZERO));
        assert_eq!(pending.len(), 1);
        assert!(pending.take("t2").is_some());
        assert!(pending.take("t1").is_none(), "taken together with t2");
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn old_parked_turns_expire() {
        let pending = PendingTurns::default();
        pending.park(parked(&["old"], Duration::from_secs(3600)));
        pending.park(parked(&["new"], Duration::ZERO));
        let expired = pending.expire(Duration::from_secs(1800));
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].tool_ids, ["old"]);
        assert!(pending.take("new").is_some());
    }
}
