use axum::Router;
use axum::routing::{get, post};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use crate::routes;
use crate::session::SessionStore;
use crate::status::RuntimeStatus;

#[derive(Clone)]
pub struct AppState {
    /// Working directory of the CLI processes; their sessions are saved under it.
    pub cwd: String,
    pub sessions: SessionStore,
    pub status: Arc<RuntimeStatus>,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .route("/v1/messages", post(routes::messages))
        .fallback(routes::fallback)
        .layer(CorsLayer::permissive())
        // Base64 images make request bodies large.
        .layer(axum::extract::DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state)
}
