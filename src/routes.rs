use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::header;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use std::convert::Infallible;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::info;

use crate::adapter::{anthropic_to_cli, cli_to_anthropic, cli_to_openai, openai_to_cli};
use crate::error::AppError;
use crate::models::{self, ALIASES};
use crate::server::AppState;
use crate::status::unix_now;
use crate::turn::{self, TurnEvent, TurnOutput, TurnRequest};
use crate::types::anthropic::MessagesRequest;
use crate::types::openai::{ChatCompletionRequest, ModelInfo, ModelsResponse};

/// How long a streaming response may hold its headers while waiting for the
/// first token. Errors that come before it (unknown model, exhausted limit,
/// bad image) then reach the client as a real HTTP status, not as an event
/// inside a 200 stream.
const HEADER_WAIT: Duration = Duration::from_secs(10);

fn generate_request_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..8].to_string()
}

pub async fn health(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "uptime": state.status.uptime_secs(),
        "cli_version": state.status.cli_version(),
        "workdir": state.cwd,
        "saved_sessions": state.sessions.len().await,
        // Turns waiting for a client to run tools and send the results.
        "waiting_for_tools": state.pending.len(),
        "models": state.status.aliases(),
        // What the subscription has used, as of the last turn; null before it.
        "rate_limits": state.status.rate_limits(),
    }))
}

/// The aliases, plus every full model id a turn has run on. Limits appear
/// once a turn has reported them.
pub async fn models(State(state): State<AppState>) -> impl IntoResponse {
    let aliases = state.status.aliases();
    let limits = state.status.models();
    let created = unix_now();
    let info = |id: &str, facts: Option<&crate::status::ModelLimits>| ModelInfo {
        id: id.to_string(),
        object: "model".to_string(),
        owned_by: "anthropic".to_string(),
        created,
        context_window: facts.and_then(|f| f.context_window),
        max_tokens: facts.and_then(|f| f.max_output_tokens),
    };

    let mut data: Vec<ModelInfo> = ALIASES
        .iter()
        .map(|alias| info(alias, aliases.get(*alias).and_then(|id| limits.get(id))))
        .collect();
    data.extend(limits.iter().map(|(id, facts)| info(id, Some(facts))));

    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
}

// ── OpenAI Chat Completions ─────────────────────────────────────

pub async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Result<Response, AppError> {
    let Json(request) = payload.map_err(|e| AppError::BadRequest(e.body_text()))?;
    let conversation = openai_to_cli::to_conversation(&request).map_err(AppError::BadRequest)?;
    let model = models::resolve(request.model.as_deref()).map_err(AppError::BadRequest)?;
    let request_id = generate_request_id();

    info!(
        "[req={request_id}] OpenAI chat model={model} stream={} turns={} tools={}",
        request.stream,
        conversation.turns().len(),
        conversation.tools.len()
    );

    let turn = TurnRequest {
        request_id: request_id.clone(),
        api: "openai",
        model: model.clone(),
        conversation,
    };
    let mut events = turn::start(&state, turn);

    if !request.stream {
        let output = collect(&request_id, events).await?;
        let body = Json(cli_to_openai::completion(&output, &request_id));
        return Ok(with_request_id(&request_id, body));
    }

    let first = first_events(&mut events).await?;
    let include_usage = request.stream_options.is_some_and(|o| o.include_usage);
    let mut writer = cli_to_openai::OpenAiStream::new(&request_id, &model, include_usage);
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let send = |payloads: Vec<String>| {
            let tx = tx.clone();
            async move {
                for payload in payloads {
                    if tx.send(Ok(Event::default().data(payload))).await.is_err() {
                        return false;
                    }
                }
                true
            }
        };
        for event in first {
            if !send(writer.on_event(event)).await {
                return;
            }
        }
        while let Some(event) = events.recv().await {
            if !send(writer.on_event(event)).await {
                return;
            }
        }
        send(writer.on_end()).await;
    });
    Ok(sse_response(&request_id, rx))
}

// ── Anthropic Messages ──────────────────────────────────────────

/// Errors on this route use Anthropic's error shape.
pub async fn messages(
    State(state): State<AppState>,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Response {
    match handle_messages(state, payload).await {
        Ok(response) => response,
        Err(e) => e.into_anthropic_response(),
    }
}

async fn handle_messages(
    state: AppState,
    payload: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Response, AppError> {
    let Json(request) = payload.map_err(|e| AppError::BadRequest(e.body_text()))?;
    let conversation = anthropic_to_cli::to_conversation(&request).map_err(AppError::BadRequest)?;
    let model = models::resolve(request.model.as_deref()).map_err(AppError::BadRequest)?;
    let request_id = generate_request_id();

    info!(
        "[req={request_id}] Anthropic messages model={model} stream={} turns={} tools={}",
        request.stream,
        conversation.turns().len(),
        conversation.tools.len()
    );

    let turn = TurnRequest {
        request_id: request_id.clone(),
        api: "anthropic",
        model: model.clone(),
        conversation,
    };
    let mut events = turn::start(&state, turn);

    if !request.stream {
        let output = collect(&request_id, events).await?;
        let body = Json(cli_to_anthropic::message(&output, &request_id));
        return Ok(with_request_id(&request_id, body));
    }

    let first = first_events(&mut events).await?;
    let mut writer = cli_to_anthropic::AnthropicStream::new(&request_id, &model);
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let send = |named: Vec<cli_to_anthropic::SseEvent>| {
            let tx = tx.clone();
            async move {
                for (name, data) in named {
                    if tx.send(Ok(Event::default().event(name).data(data))).await.is_err() {
                        return false;
                    }
                }
                true
            }
        };
        for event in first {
            if !send(writer.on_event(event)).await {
                return;
            }
        }
        while let Some(event) = events.recv().await {
            if !send(writer.on_event(event)).await {
                return;
            }
        }
        send(writer.on_end()).await;
    });
    Ok(sse_response(&request_id, rx))
}

// ── Shared ──────────────────────────────────────────────────────

/// Wait for the end of a non-streaming turn.
async fn collect(request_id: &str, mut events: mpsc::Receiver<TurnEvent>) -> Result<TurnOutput, AppError> {
    let start = Instant::now();
    while let Some(event) = events.recv().await {
        match event {
            TurnEvent::Finished(output) => {
                info!("[req={request_id}] Complete after {:.2}s", start.elapsed().as_secs_f64());
                return Ok(output);
            }
            TurnEvent::Failed(e) => {
                info!("[req={request_id}] Failed after {:.2}s", start.elapsed().as_secs_f64());
                return Err(AppError::upstream(e));
            }
            TurnEvent::Started { .. } | TurnEvent::Delta(_) => {}
        }
    }
    Err(AppError::Internal("the turn ended without a result".to_string()))
}

/// Events up to the first token, the end of the turn, or `HEADER_WAIT`,
/// whichever comes first. A failure in that window becomes the HTTP response.
async fn first_events(events: &mut mpsc::Receiver<TurnEvent>) -> Result<Vec<TurnEvent>, AppError> {
    let deadline = tokio::time::Instant::now() + HEADER_WAIT;
    let mut seen = Vec::new();
    loop {
        match tokio::time::timeout_at(deadline, events.recv()).await {
            Ok(Some(TurnEvent::Failed(e))) => return Err(AppError::upstream(e)),
            Ok(Some(event @ TurnEvent::Started { .. })) => seen.push(event),
            Ok(Some(event)) => {
                seen.push(event);
                return Ok(seen);
            }
            Ok(None) => return Err(AppError::Internal("the turn ended without a result".to_string())),
            Err(_) => return Ok(seen),
        }
    }
}

fn sse_response(request_id: &str, rx: mpsc::Receiver<Result<Event, Infallible>>) -> Response {
    let sse = Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default());
    (
        [
            (header::HeaderName::from_static("x-request-id"), request_id.to_string()),
            (header::CACHE_CONTROL, "no-cache".to_string()),
        ],
        sse,
    )
        .into_response()
}

fn with_request_id(request_id: &str, body: impl IntoResponse) -> Response {
    (
        [(header::HeaderName::from_static("x-request-id"), request_id.to_string())],
        body,
    )
        .into_response()
}

pub async fn fallback() -> impl IntoResponse {
    AppError::NotFound("The requested endpoint does not exist".to_string())
}
