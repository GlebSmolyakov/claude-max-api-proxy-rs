mod adapter;
mod conversation;
mod error;
mod models;
mod routes;
mod server;
mod session;
mod status;
mod subprocess;
mod turn;
mod types;

use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "claude-max-api")]
#[command(about = "OpenAI & Anthropic-compatible API proxy for Claude Code CLI")]
struct Args {
    /// Port to listen on
    #[arg(default_value = "8080")]
    port: u16,

    /// Working directory for the Claude CLI processes
    /// [default: ~/.claude-max-api/workdir]
    #[arg(long = "cwd")]
    cwd: Option<PathBuf>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing with compact format
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "claude_max_api=info".parse().unwrap()),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .compact()
        .init();

    let args = Args::parse();

    // State lives in ~/.claude-max-api: the session map, and by default the
    // working directory whose CLI sessions the proxy owns.
    let state_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".claude-max-api");
    let cwd = args.cwd.unwrap_or_else(|| state_dir.join("workdir"));
    if let Err(e) = std::fs::create_dir_all(&cwd) {
        error!("Cannot create {}: {e}", cwd.display());
        std::process::exit(1);
    }
    let cwd = std::fs::canonicalize(&cwd).unwrap_or(cwd);

    // Verify claude CLI is available
    let cli_version = match tokio::process::Command::new("claude")
        .arg("--version")
        .output()
        .await
    {
        Ok(output) => {
            let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
            info!("Found claude CLI: {}", version);
            version
        }
        Err(e) => {
            error!("claude CLI not found: {}. Install it with: npm install -g @anthropic-ai/claude-code", e);
            std::process::exit(1);
        }
    };

    let sessions = session::SessionStore::open(
        state_dir.join("sessions.json"),
        session::transcripts_dir_for(&cwd),
    )
    .await;
    sessions.spawn_cleanup_task();

    let state = server::AppState {
        cwd: cwd.to_string_lossy().to_string(),
        sessions,
        status: Arc::new(status::RuntimeStatus::new(cli_version)),
    };

    let app = server::create_router(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind to {}: {}", addr, e);
            if e.kind() == std::io::ErrorKind::AddrInUse {
                error!("Port {} is already in use", args.port);
            }
            std::process::exit(1);
        }
    };

    info!("claude-max-proxy listening on http://127.0.0.1:{} (cwd: {})", args.port, cwd.display());
    info!("endpoints: GET /health, /v1/models | POST /v1/chat/completions (OpenAI), /v1/messages (Anthropic)");

    // Graceful shutdown on SIGINT/SIGTERM
    let shutdown = async {
        let ctrl_c = tokio::signal::ctrl_c();
        #[cfg(unix)]
        {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = ctrl_c => { info!("Received SIGINT, shutting down..."); }
                _ = sigterm.recv() => { info!("Received SIGTERM, shutting down..."); }
            }
        }
        #[cfg(not(unix))]
        {
            ctrl_c.await.ok();
            info!("Received SIGINT, shutting down...");
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .unwrap_or_else(|e| {
            error!("Server error: {}", e);
            std::process::exit(1);
        });

    info!("Server stopped.");
}
