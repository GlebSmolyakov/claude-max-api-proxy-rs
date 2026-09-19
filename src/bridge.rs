//! Lends the client's tools to the CLI through a tiny MCP server.
//!
//! When a request declares tools, its CLI process gets `--mcp-config`
//! pointing at `/mcp/<token>` on this proxy. The CLI lists the tools here,
//! and when the model calls one, it sends `tools/call` here. That call waits
//! until the client, the only one that can actually run the tool, sends the
//! result with its next request; the turn runner then hands it over with
//! `Bridge::deliver`.

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use tracing::{debug, warn};

use crate::conversation::{Block, ImageSource, ToolDef};
use crate::server::AppState;
use crate::subprocess::MCP_SERVER;

/// What a client tool returned.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutcome {
    /// Text and images.
    pub content: Vec<Block>,
    pub is_error: bool,
}

enum Slot {
    /// The CLI is waiting for this result.
    Waiting(oneshot::Sender<ToolOutcome>),
    /// The result came before the CLI asked for it.
    Ready(ToolOutcome),
}

pub struct Bridge {
    tools: Vec<ToolDef>,
    calls: Mutex<HashMap<String, Slot>>,
}

impl Bridge {
    pub fn new(tools: Vec<ToolDef>) -> Self {
        Self {
            tools,
            calls: Mutex::new(HashMap::new()),
        }
    }

    /// Hand the result of tool call `tool_use_id` to the CLI.
    pub fn deliver(&self, tool_use_id: &str, outcome: ToolOutcome) {
        let mut calls = self.calls.lock().unwrap();
        match calls.remove(tool_use_id) {
            Some(Slot::Waiting(tx)) => {
                let _ = tx.send(outcome);
            }
            _ => {
                calls.insert(tool_use_id.to_string(), Slot::Ready(outcome));
            }
        }
    }

    /// Wait for the result of `tool_use_id`, which may already be here.
    fn wait(&self, tool_use_id: &str) -> oneshot::Receiver<ToolOutcome> {
        let (tx, rx) = oneshot::channel();
        let mut calls = self.calls.lock().unwrap();
        match calls.remove(tool_use_id) {
            Some(Slot::Ready(outcome)) => {
                let _ = tx.send(outcome);
            }
            _ => {
                calls.insert(tool_use_id.to_string(), Slot::Waiting(tx));
            }
        }
        rx
    }

    /// The client's name for a tool the model called as `mcp__c__<name>`.
    /// The CLI may have replaced characters it does not allow in tool
    /// names, so a name that does not match exactly is looked up that way.
    pub fn client_name(&self, model_name: &str) -> String {
        let prefix = format!("mcp__{MCP_SERVER}__");
        let name = model_name.strip_prefix(&prefix).unwrap_or(model_name);
        if self.tools.iter().any(|t| t.name == name) {
            return name.to_string();
        }
        self.tools
            .iter()
            .find(|t| sanitize(&t.name) == name)
            .map_or_else(|| name.to_string(), |t| t.name.clone())
    }

    /// Answer one JSON-RPC message. `None` for notifications.
    async fn handle(&self, message: &Value) -> Option<Value> {
        let id = message.get("id")?.clone();
        let method = message.get("method").and_then(Value::as_str).unwrap_or("");
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        let result = match method {
            "initialize" => json!({
                "protocolVersion": params.get("protocolVersion").cloned().unwrap_or(json!("2025-06-18")),
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "claude-max-api", "version": env!("CARGO_PKG_VERSION") },
            }),
            "ping" => json!({}),
            "tools/list" => json!({ "tools": self.tools.iter().map(tool_json).collect::<Vec<_>>() }),
            "tools/call" => self.call(&params).await,
            _ => {
                return Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": format!("method not found: {method}") },
                }));
            }
        };
        Some(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
    }

    async fn call(&self, params: &Value) -> Value {
        // The CLI names the tool call it is running; that id is the one the
        // client got and will answer to.
        let Some(tool_use_id) = params
            .pointer("/_meta/claudecode~1toolUseId")
            .and_then(Value::as_str)
        else {
            warn!("tools/call without claudecode/toolUseId: {params}");
            return error_result("the proxy could not tell which tool call this is");
        };
        debug!("Tool call {tool_use_id} is waiting for the client");
        match self.wait(tool_use_id).await {
            Ok(outcome) => json!({
                "content": outcome_content(&outcome.content),
                "isError": outcome.is_error,
            }),
            Err(_) => error_result("the turn ended before the client returned this tool's result"),
        }
    }
}

/// Characters other than letters, digits, `_` and `-` become `_`.
fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

fn tool_json(tool: &ToolDef) -> Value {
    json!({ "name": tool.name, "description": tool.description, "inputSchema": tool.input_schema })
}

fn error_result(message: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": message }], "isError": true })
}

fn outcome_content(blocks: &[Block]) -> Vec<Value> {
    let content: Vec<Value> = blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text(text) => Some(json!({ "type": "text", "text": text })),
            Block::Image(ImageSource::Base64 { media_type, data }) => {
                Some(json!({ "type": "image", "data": data, "mimeType": media_type }))
            }
            Block::Image(ImageSource::Url(url)) => Some(json!({ "type": "text", "text": format!("[image: {url}]") })),
            Block::ToolUse(_) | Block::ToolResult { .. } => None,
        })
        .collect();
    if content.is_empty() {
        vec![json!({ "type": "text", "text": "(no output)" })]
    } else {
        content
    }
}

/// The bridges of running turns, by token.
#[derive(Clone, Default)]
pub struct Bridges {
    inner: Arc<Mutex<HashMap<String, Arc<Bridge>>>>,
}

impl Bridges {
    pub fn register(&self, tools: Vec<ToolDef>) -> (String, Arc<Bridge>) {
        let token = uuid::Uuid::new_v4().simple().to_string();
        let bridge = Arc::new(Bridge::new(tools));
        self.inner.lock().unwrap().insert(token.clone(), bridge.clone());
        (token, bridge)
    }

    pub fn get(&self, token: &str) -> Option<Arc<Bridge>> {
        self.inner.lock().unwrap().get(token).cloned()
    }

    pub fn remove(&self, token: &str) {
        self.inner.lock().unwrap().remove(token);
    }
}

// ── MCP Streamable HTTP endpoint ────────────────────────────────

/// JSON-RPC over POST, answered with plain JSON. A batch gets an array.
pub async fn mcp_post(State(state): State<AppState>, Path(token): Path<String>, body: Bytes) -> Response {
    let Some(bridge) = state.bridges.get(&token) else {
        return (StatusCode::NOT_FOUND, "unknown MCP session").into_response();
    };
    let message: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let error = json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": e.to_string() } });
            return (StatusCode::BAD_REQUEST, axum::Json(error)).into_response();
        }
    };
    let reply = match &message {
        Value::Array(batch) => {
            let mut replies = Vec::new();
            for m in batch {
                replies.extend(bridge.handle(m).await);
            }
            (!replies.is_empty()).then_some(Value::Array(replies))
        }
        single => bridge.handle(single).await,
    };
    match reply {
        Some(reply) => axum::Json(reply).into_response(),
        None => StatusCode::ACCEPTED.into_response(),
    }
}

/// No server-initiated messages: the optional SSE stream is not offered.
pub async fn mcp_get() -> StatusCode {
    StatusCode::METHOD_NOT_ALLOWED
}

pub async fn mcp_delete() -> StatusCode {
    StatusCode::OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bridge() -> Bridge {
        Bridge::new(vec![
            ToolDef {
                name: "read_file".into(),
                description: "Read a file".into(),
                input_schema: json!({ "type": "object", "properties": { "path": { "type": "string" } } }),
            },
            ToolDef { name: "fs.list".into(), description: String::new(), input_schema: json!({ "type": "object" }) },
        ])
    }

    fn call_message(id: u64, tool_use_id: &str) -> Value {
        json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": { "name": "read_file", "arguments": { "path": "a.rs" }, "_meta": { "claudecode/toolUseId": tool_use_id } },
        })
    }

    fn text_outcome(text: &str) -> ToolOutcome {
        ToolOutcome { content: vec![Block::Text(text.into())], is_error: false }
    }

    #[tokio::test]
    async fn initialize_echoes_the_protocol_version() {
        let reply = bridge()
            .handle(&json!({ "jsonrpc": "2.0", "id": 0, "method": "initialize", "params": { "protocolVersion": "2025-11-25" } }))
            .await
            .unwrap();
        assert_eq!(reply["result"]["protocolVersion"], "2025-11-25");
        assert_eq!(reply["result"]["capabilities"]["tools"], json!({}));
    }

    #[tokio::test]
    async fn notifications_get_no_reply_and_unknown_methods_an_error() {
        let b = bridge();
        assert!(b.handle(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).await.is_none());
        let reply = b.handle(&json!({ "jsonrpc": "2.0", "id": 1, "method": "server/discover" })).await.unwrap();
        assert_eq!(reply["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn tools_list_carries_the_client_tools() {
        let reply = bridge().handle(&json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" })).await.unwrap();
        let tools = reply["result"]["tools"].as_array().unwrap();
        assert_eq!(tools[0]["name"], "read_file");
        assert_eq!(tools[0]["inputSchema"]["properties"]["path"]["type"], "string");
    }

    #[tokio::test]
    async fn a_call_waits_for_the_client_result() {
        let b = Arc::new(bridge());
        let waiting = {
            let b = b.clone();
            tokio::spawn(async move { b.handle(&call_message(7, "toolu_1")).await.unwrap() })
        };
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished(), "no result yet");
        b.deliver("toolu_1", text_outcome("fn main() {}"));
        let reply = waiting.await.unwrap();
        assert_eq!(reply["id"], 7);
        assert_eq!(reply["result"]["content"][0]["text"], "fn main() {}");
        assert_eq!(reply["result"]["isError"], false);
    }

    #[tokio::test]
    async fn a_result_that_arrives_first_is_kept() {
        let b = bridge();
        b.deliver("toolu_2", ToolOutcome { content: vec![Block::Text("boom".into())], is_error: true });
        let reply = b.handle(&call_message(1, "toolu_2")).await.unwrap();
        assert_eq!(reply["result"]["isError"], true);
    }

    #[tokio::test]
    async fn a_call_without_an_id_fails_at_once() {
        let reply = bridge()
            .handle(&json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": { "name": "read_file", "arguments": {} } }))
            .await
            .unwrap();
        assert_eq!(reply["result"]["isError"], true);
    }

    #[test]
    fn images_become_mcp_images_and_empty_output_is_marked() {
        let content = outcome_content(&[
            Block::Image(ImageSource::Base64 { media_type: "image/png".into(), data: "QQ==".into() }),
        ]);
        assert_eq!(content[0], json!({ "type": "image", "data": "QQ==", "mimeType": "image/png" }));
        assert_eq!(outcome_content(&[])[0]["text"], "(no output)");
    }

    #[test]
    fn client_names_lose_the_prefix_and_survive_sanitizing() {
        let b = bridge();
        assert_eq!(b.client_name("mcp__c__read_file"), "read_file");
        assert_eq!(b.client_name("mcp__c__fs_list"), "fs.list");
        assert_eq!(b.client_name("mcp__c__unknown"), "unknown");
    }

    #[test]
    fn registry_hands_out_distinct_tokens() {
        let bridges = Bridges::default();
        let (a, _) = bridges.register(vec![]);
        let (b, _) = bridges.register(vec![]);
        assert_ne!(a, b);
        assert!(bridges.get(&a).is_some());
        bridges.remove(&a);
        assert!(bridges.get(&a).is_none());
    }
}
