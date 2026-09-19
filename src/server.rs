use axum::Router;
use axum::routing::{get, post};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::bridge::{self, Bridges};
use crate::routes;
use crate::session::SessionStore;
use crate::status::RuntimeStatus;
use crate::turn::PendingTurns;

#[derive(Clone)]
pub struct AppState {
    /// Working directory of the CLI processes; their sessions are saved under it.
    pub cwd: String,
    pub sessions: SessionStore,
    pub status: Arc<RuntimeStatus>,
    /// MCP servers that lend client tools to running CLI processes.
    pub bridges: Bridges,
    /// Turns waiting for the client to run tools.
    pub pending: Arc<PendingTurns>,
    /// `http://127.0.0.1:<port>/mcp`, the base of the bridge URLs.
    pub mcp_base: String,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/messages", post(routes::messages))
        .route(
            "/mcp/{token}",
            post(bridge::mcp_post).get(bridge::mcp_get).delete(bridge::mcp_delete),
        )
        .fallback(routes::fallback)
        .layer(CorsLayer::permissive())
        // Base64 images make request bodies large.
        .layer(axum::extract::DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}
