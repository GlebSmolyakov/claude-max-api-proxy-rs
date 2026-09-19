use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::turn::TurnError;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Internal error: {0}")]
    Internal(String),

    /// The CLI or the API behind it failed; `status` says how.
    #[error("Upstream error ({status}): {message}")]
    Upstream { status: StatusCode, message: String },
}

impl AppError {
    pub fn upstream(error: TurnError) -> Self {
        AppError::Upstream {
            status: StatusCode::from_u16(error.status).unwrap_or(StatusCode::BAD_GATEWAY),
            message: error.message,
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::Upstream { status, .. } => *status,
        }
    }

    fn message(&self) -> &str {
        match self {
            AppError::BadRequest(m) | AppError::NotFound(m) | AppError::Internal(m) => m,
            AppError::Upstream { message, .. } => message,
        }
    }

    /// `{"error": {"message", "type", "code"}}`, as OpenAI returns it.
    pub fn openai_body(&self) -> Value {
        let status = self.status();
        let error_type = match status.as_u16() {
            429 => "rate_limit_error",
            401 | 403 => "authentication_error",
            504 => "timeout",
            400..=499 => "invalid_request_error",
            _ => "server_error",
        };
        let code = match self {
            AppError::NotFound(_) => Some("not_found"),
            _ => None,
        };
        json!({ "error": { "message": self.message(), "type": error_type, "code": code } })
    }

    /// `{"type": "error", "error": {"type", "message"}}`, as Anthropic returns it.
    pub fn anthropic_body(&self) -> Value {
        let error_type = match self.status().as_u16() {
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            529 => "overloaded_error",
            400..=499 => "invalid_request_error",
            _ => "api_error",
        };
        json!({ "type": "error", "error": { "type": error_type, "message": self.message() } })
    }

    pub fn into_anthropic_response(self) -> Response {
        (self.status(), axum::Json(self.anthropic_body())).into_response()
    }
}

/// Errors render in the OpenAI shape unless a route asks for Anthropic's.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status(), axum::Json(self.openai_body())).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn upstream(status: u16) -> AppError {
        AppError::upstream(TurnError { status, message: "boom".into() })
    }

    #[test]
    fn statuses() {
        assert_eq!(AppError::BadRequest("x".into()).status(), StatusCode::BAD_REQUEST);
        assert_eq!(AppError::NotFound("x".into()).status(), StatusCode::NOT_FOUND);
        assert_eq!(AppError::Internal("x".into()).status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(upstream(429).status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(upstream(0).status(), StatusCode::BAD_GATEWAY, "an invalid code falls back to 502");
    }

    #[test]
    fn openai_shape() {
        let body = upstream(429).openai_body();
        assert_eq!(body["error"]["type"], "rate_limit_error");
        assert_eq!(body["error"]["message"], "boom");
        assert_eq!(AppError::BadRequest("bad".into()).openai_body()["error"]["type"], "invalid_request_error");
        assert_eq!(AppError::NotFound("x".into()).openai_body()["error"]["code"], "not_found");
    }

    #[test]
    fn anthropic_shape() {
        let body = upstream(529).anthropic_body();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "overloaded_error");
        assert_eq!(upstream(502).anthropic_body()["error"]["type"], "api_error");
        assert_eq!(AppError::BadRequest("x".into()).anthropic_body()["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn responses_carry_the_status() {
        assert_eq!(upstream(400).into_response().status(), StatusCode::BAD_REQUEST);
        assert_eq!(upstream(529).into_anthropic_response().status().as_u16(), 529);
    }
}
