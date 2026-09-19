//! Runs one `claude --print` process per turn and turns its NDJSON output
//! into events.

use crate::conversation::ToolCall;
use crate::types::claude_cli::{
    AssistantBlock, ClaudeCliMessage, RateLimitInfo, ResultMessage, ResultUsage, StartedBlock, StreamEvent,
};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

/// Also bounds how long a turn may wait for the client to run a tool.
pub const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const STDERR_TAIL_LINES: usize = 5;

/// Environment that keeps the CLI a plain model instead of the user's agent.
pub const CLEAN_ENV: [(&str, &str); 8] = [
    // No CLAUDE.md or memory files in the context.
    ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "1"),
    ("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1"),
    // In -p mode this skips an extra model call that titles every session.
    ("CLAUDE_CODE_DISABLE_TERMINAL_TITLE", "1"),
    // No update, telemetry or feature-flag calls at startup: about 1 s saved per turn.
    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
    // Client tools arrive through an MCP server. They must be in the prompt
    // from the first call, not deferred behind tool search, and the tool
    // list must come from this process's server, never from a cache.
    ("ENABLE_TOOL_SEARCH", "false"),
    ("MCP_DISCOVERY_CACHE", "0"),
    ("MCP_CONNECTION_NONBLOCKING", "0"),
    // Tool results such as whole files can be large.
    ("MAX_MCP_OUTPUT_TOKENS", "100000"),
];

/// Name of the MCP server that carries client tools. The model sees each
/// tool as `mcp__c__<name>`.
pub const MCP_SERVER: &str = "c";

#[derive(Debug)]
pub enum SubprocessEvent {
    /// The session started; `model` is the id the alias resolved to.
    Init { session_id: String, model: String },
    TextDelta(String),
    /// A tool call began; its input follows in `ToolUse`.
    ToolUseStarted(String),
    /// A complete tool call. `name` still has the MCP prefix.
    ToolUse(ToolCall),
    /// One API call finished.
    StepEnd { stop_reason: Option<String>, usage: Option<ResultUsage> },
    RateLimit(RateLimitInfo),
    Result(ResultMessage),
    /// The process could not be started or read.
    Error(String),
    /// No output for `INACTIVITY_TIMEOUT`; the process was killed.
    Timeout,
    Close { code: i32, stderr_tail: String },
}

#[derive(Debug, Clone)]
pub struct SubprocessOptions {
    pub request_id: String,
    pub model: String,
    pub system_prompt: String,
    /// Continue this saved session, forking it.
    pub resume: Option<String>,
    /// URL of the MCP server with the client's tools, when there are any.
    pub mcp_url: Option<String>,
    pub cwd: String,
    pub api: &'static str,
}

/// A running CLI process. It lives as long as this value: dropping it kills
/// the process, which is how an abandoned turn gets cleaned up.
pub struct Process {
    pub events: mpsc::Receiver<SubprocessEvent>,
    _alive: oneshot::Sender<()>,
}

impl Process {
    /// A process whose events come from `events`, for tests.
    #[cfg(test)]
    pub fn from_events(events: mpsc::Receiver<SubprocessEvent>) -> Self {
        let (alive, _) = oneshot::channel();
        Self { events, _alive: alive }
    }
}

/// Start the CLI and write `input` (one NDJSON user message) to its stdin.
pub fn spawn(input: String, options: SubprocessOptions) -> Process {
    let (tx, events) = mpsc::channel(64);
    let (alive, dropped) = oneshot::channel();
    tokio::spawn(run(input, options, tx, dropped));
    Process { events, _alive: alive }
}

pub fn build_args(options: &SubprocessOptions) -> Vec<String> {
    let mut args: Vec<String> = [
        "--print",
        "--verbose",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        // A plain model, not a coding agent: no built-in tools, no MCP
        // servers, no skills, no settings files from this machine.
        "--tools",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--setting-sources",
        "",
    ]
    .map(String::from)
    .to_vec();
    args.extend([
        "--model".to_string(),
        options.model.clone(),
        "--system-prompt".to_string(),
        options.system_prompt.clone(),
    ]);
    if let Some(url) = &options.mcp_url {
        args.extend([
            "--mcp-config".to_string(),
            mcp_config(url),
            "--allowedTools".to_string(),
            format!("mcp__{MCP_SERVER}"),
        ]);
    }
    if let Some(session_id) = &options.resume {
        args.extend([
            "--resume".to_string(),
            session_id.clone(),
            "--fork-session".to_string(),
        ]);
    }
    args
}

/// `--mcp-config` for the client-tools server. `timeout` lifts the CLI's
/// default 60 s limit per HTTP request: a call waits while the client runs
/// the tool.
pub fn mcp_config(url: &str) -> String {
    serde_json::json!({
        "mcpServers": {
            MCP_SERVER: {
                "type": "http",
                "url": url,
                "alwaysLoad": true,
                "timeout": INACTIVITY_TIMEOUT.as_millis() as u64,
            }
        }
    })
    .to_string()
}

async fn run(
    input: String,
    options: SubprocessOptions,
    tx: mpsc::Sender<SubprocessEvent>,
    mut dropped: oneshot::Receiver<()>,
) {
    let rid = options.request_id.clone();
    let start = Instant::now();
    let mode = if options.resume.is_some() { "resume" } else { "fresh" };
    let tools = if options.mcp_url.is_some() { " tools=on" } else { "" };
    info!("[req={rid}] Spawning claude model={} api={} session={mode}{tools}", options.model, options.api);

    let mut command = Command::new("claude");
    command
        .args(build_args(&options))
        .current_dir(&options.cwd)
        .envs(CLEAN_ENV)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            let msg = if e.kind() == std::io::ErrorKind::NotFound {
                "claude CLI not found. Install it with: npm install -g @anthropic-ai/claude-code".to_string()
            } else {
                format!("Failed to spawn claude: {e}")
            };
            error!("[req={rid}] {msg}");
            let _ = tx.send(SubprocessEvent::Error(msg)).await;
            return;
        }
    };
    let pid = child.id().unwrap_or(0);

    // The prompt goes through stdin: no argument-length limits, no quoting.
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let written = stdin.write_all(input.as_bytes()).await;
    drop(stdin); // closing stdin tells the CLI the input is complete
    if let Err(e) = written {
        error!("[req={rid}][pid={pid}] Failed to write the prompt: {e}");
        let _ = child.kill().await;
        let _ = tx.send(SubprocessEvent::Error(format!("stdin write failed: {e}"))).await;
        return;
    }

    let mut stdout = BufReader::new(child.stdout.take().expect("stdout is piped")).lines();
    let mut stderr = BufReader::new(child.stderr.take().expect("stderr is piped")).lines();
    let mut stderr_tail: VecDeque<String> = VecDeque::new();
    let mut stderr_open = true;
    let mut ttft: Option<f64> = None;
    let mut chunks: u64 = 0;
    // Lines since the last progress report; an idle process, such as one
    // parked while the client runs tools, reports nothing.
    let mut recent_lines: u64 = 0;

    let inactivity = tokio::time::sleep(INACTIVITY_TIMEOUT);
    tokio::pin!(inactivity);
    let progress = tokio::time::sleep(Duration::from_secs(30));
    tokio::pin!(progress);

    loop {
        tokio::select! {
            line = stdout.next_line() => {
                let line = match line {
                    Ok(Some(line)) => line,
                    Ok(None) => break,
                    Err(e) => {
                        error!("[req={rid}][pid={pid}] Error reading stdout: {e}");
                        break;
                    }
                };
                inactivity.as_mut().reset(tokio::time::Instant::now() + INACTIVITY_TIMEOUT);
                recent_lines += 1;
                for event in process_line(&line) {
                    if matches!(event, SubprocessEvent::TextDelta(_)) {
                        chunks += 1;
                        if ttft.is_none() {
                            let t = start.elapsed().as_secs_f64();
                            info!("[req={rid}][pid={pid}] First token after {t:.2}s");
                            ttft = Some(t);
                        }
                    }
                    if tx.send(event).await.is_err() {
                        warn!("[req={rid}][pid={pid}] Nobody is listening after {:.2}s, killing claude", start.elapsed().as_secs_f64());
                        let _ = child.kill().await;
                        return;
                    }
                }
            }
            line = stderr.next_line(), if stderr_open => {
                match line {
                    Ok(Some(line)) => {
                        inactivity.as_mut().reset(tokio::time::Instant::now() + INACTIVITY_TIMEOUT);
                        debug!("[req={rid}][pid={pid}] stderr: {line}");
                        if !line.trim().is_empty() {
                            if stderr_tail.len() == STDERR_TAIL_LINES {
                                stderr_tail.pop_front();
                            }
                            stderr_tail.push_back(line);
                        }
                    }
                    _ => stderr_open = false,
                }
            }
            _ = &mut dropped => {
                info!("[req={rid}][pid={pid}] Turn abandoned after {:.0}s, killing claude", start.elapsed().as_secs_f64());
                let _ = child.kill().await;
                return;
            }
            () = &mut progress => {
                if recent_lines > 0 {
                    info!("[req={rid}][pid={pid}] Still running {:.0}s chunks={chunks}", start.elapsed().as_secs_f64());
                    recent_lines = 0;
                }
                progress.as_mut().reset(tokio::time::Instant::now() + Duration::from_secs(30));
            }
            () = &mut inactivity => {
                warn!("[req={rid}][pid={pid}] No output for 30 minutes, killing claude");
                let _ = child.kill().await;
                let _ = tx.send(SubprocessEvent::Timeout).await;
                return;
            }
        }
    }

    // Collect what is left on stderr, since errors are often printed right
    // before exit. Bounded, in case something keeps the pipe open.
    let drain = async {
        while stderr_open {
            match stderr.next_line().await {
                Ok(Some(line)) if !line.trim().is_empty() => {
                    if stderr_tail.len() == STDERR_TAIL_LINES {
                        stderr_tail.pop_front();
                    }
                    stderr_tail.push_back(line);
                }
                Ok(Some(_)) => {}
                _ => stderr_open = false,
            }
        }
    };
    let _ = tokio::time::timeout(Duration::from_secs(2), drain).await;

    let code = match child.wait().await {
        Ok(status) => status.code().unwrap_or(-1),
        Err(e) => {
            error!("[req={rid}][pid={pid}] Error waiting for claude: {e}");
            -1
        }
    };
    let ttft = ttft.map_or("-".to_string(), |t| format!("{t:.2}s"));
    info!(
        "[req={rid}][pid={pid}] Done model={} ttft={ttft} total={:.2}s exit={code}",
        options.model,
        start.elapsed().as_secs_f64()
    );
    let stderr_tail = Vec::from(stderr_tail).join("\n");
    let _ = tx.send(SubprocessEvent::Close { code, stderr_tail }).await;
}

/// Parse one stdout line. Lines the proxy does not use yield nothing.
fn process_line(line: &str) -> Vec<SubprocessEvent> {
    let line = line.trim();
    if line.is_empty() {
        return vec![];
    }
    let message = match serde_json::from_str::<ClaudeCliMessage>(line) {
        Ok(message) => message,
        Err(_) => {
            debug!("Ignoring line: {}", line.chars().take(200).collect::<String>());
            return vec![];
        }
    };
    match message {
        ClaudeCliMessage::System(s) if s.subtype.as_deref() == Some("init") => {
            vec![SubprocessEvent::Init {
                session_id: s.session_id.unwrap_or_default(),
                model: s.model.unwrap_or_default(),
            }]
        }
        ClaudeCliMessage::StreamEvent { event } => match event {
            StreamEvent::ContentBlockDelta { delta } => match delta.text {
                Some(text) if !text.is_empty() => vec![SubprocessEvent::TextDelta(text)],
                _ => vec![],
            },
            StreamEvent::ContentBlockStart {
                content_block: StartedBlock::ToolUse { id },
            } => vec![SubprocessEvent::ToolUseStarted(id)],
            StreamEvent::MessageDelta { delta, usage } => vec![SubprocessEvent::StepEnd {
                stop_reason: delta.stop_reason,
                usage,
            }],
            _ => vec![],
        },
        ClaudeCliMessage::Assistant { message } => message
            .content
            .into_iter()
            .filter_map(|block| match block {
                AssistantBlock::ToolUse { id, name, input } => {
                    Some(SubprocessEvent::ToolUse(ToolCall { id, name, input }))
                }
                AssistantBlock::Other => None,
            })
            .collect(),
        ClaudeCliMessage::RateLimit { rate_limit_info } => vec![SubprocessEvent::RateLimit(rate_limit_info)],
        ClaudeCliMessage::Result(result) => vec![SubprocessEvent::Result(result)],
        ClaudeCliMessage::System(_) => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(resume: Option<&str>, mcp_url: Option<&str>) -> SubprocessOptions {
        SubprocessOptions {
            request_id: "r".into(),
            model: "haiku".into(),
            system_prompt: "Be brief.".into(),
            resume: resume.map(String::from),
            mcp_url: mcp_url.map(String::from),
            cwd: "/tmp".into(),
            api: "openai",
        }
    }

    fn value_after(args: &[String], flag: &str) -> Option<String> {
        args.iter().position(|a| a == flag).map(|i| args[i + 1].clone())
    }

    #[test]
    fn args_run_a_plain_model() {
        let args = build_args(&options(None, None));
        assert_eq!(value_after(&args, "--tools").as_deref(), Some(""));
        assert_eq!(value_after(&args, "--setting-sources").as_deref(), Some(""));
        assert_eq!(value_after(&args, "--input-format").as_deref(), Some("stream-json"));
        assert_eq!(value_after(&args, "--output-format").as_deref(), Some("stream-json"));
        assert!(args.contains(&"--strict-mcp-config".to_string()));
        assert!(args.contains(&"--disable-slash-commands".to_string()));
        assert!(args.contains(&"--include-partial-messages".to_string()));
        assert!(!args.contains(&"--mcp-config".to_string()), "no tools, no MCP server");
    }

    #[test]
    fn args_carry_model_and_system_prompt() {
        let args = build_args(&options(None, None));
        assert_eq!(value_after(&args, "--model").as_deref(), Some("haiku"));
        assert_eq!(value_after(&args, "--system-prompt").as_deref(), Some("Be brief."));
    }

    #[test]
    fn fresh_sessions_do_not_resume() {
        let args = build_args(&options(None, None));
        assert!(!args.contains(&"--resume".to_string()));
        assert!(!args.contains(&"--no-session-persistence".to_string()), "sessions must be saved to be resumed");
    }

    #[test]
    fn resumed_sessions_fork() {
        let args = build_args(&options(Some("abc"), None));
        assert_eq!(value_after(&args, "--resume").as_deref(), Some("abc"));
        assert!(args.contains(&"--fork-session".to_string()));
    }

    #[test]
    fn client_tools_come_through_an_allowed_mcp_server() {
        let args = build_args(&options(None, Some("http://127.0.0.1:8080/mcp/tok")));
        let config: serde_json::Value = serde_json::from_str(&value_after(&args, "--mcp-config").unwrap()).unwrap();
        let server = &config["mcpServers"]["c"];
        assert_eq!(server["type"], "http");
        assert_eq!(server["url"], "http://127.0.0.1:8080/mcp/tok");
        assert_eq!(server["alwaysLoad"], true);
        assert!(server["timeout"].as_u64().unwrap() > 60_000, "above the CLI's 60 s default");
        assert_eq!(value_after(&args, "--allowedTools").as_deref(), Some("mcp__c"));
    }

    #[test]
    fn clean_env_disables_context_background_calls_and_tool_deferral() {
        let env: std::collections::HashMap<&str, &str> = CLEAN_ENV.into_iter().collect();
        assert_eq!(env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"], "1");
        assert_eq!(env["CLAUDE_CODE_DISABLE_TERMINAL_TITLE"], "1");
        assert_eq!(env["ENABLE_TOOL_SEARCH"], "false");
        assert_eq!(env["MCP_DISCOVERY_CACHE"], "0");
    }

    #[test]
    fn init_line_yields_session_and_model() {
        let events = process_line(r#"{"type":"system","subtype":"init","session_id":"s1","model":"claude-haiku-4-5-20251001","tools":[]}"#);
        assert!(matches!(&events[..], [SubprocessEvent::Init { session_id, model }]
            if session_id == "s1" && model == "claude-haiku-4-5-20251001"));
    }

    #[test]
    fn other_system_lines_are_ignored() {
        assert!(process_line(r#"{"type":"system","subtype":"status","status":"requesting"}"#).is_empty());
    }

    #[test]
    fn text_delta_is_forwarded_and_thinking_is_not() {
        let text = process_line(r#"{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}}"#);
        assert!(matches!(&text[..], [SubprocessEvent::TextDelta(t)] if t == "Hi"));
        let thinking = process_line(r#"{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hm"}}}"#);
        assert!(thinking.is_empty());
    }

    #[test]
    fn assistant_text_lines_are_not_forwarded() {
        assert!(process_line(r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Hi"}]}}"#).is_empty());
    }

    #[test]
    fn tool_calls_and_step_ends_are_forwarded() {
        let started = process_line(r#"{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"mcp__c__x","input":{}}}}"#);
        assert!(matches!(&started[..], [SubprocessEvent::ToolUseStarted(id)] if id == "toolu_1"));

        let call = process_line(r#"{"type":"assistant","message":{"content":[{"type":"tool_use","id":"toolu_1","name":"mcp__c__read_file","input":{"path":"a.rs"}}]}}"#);
        let [SubprocessEvent::ToolUse(call)] = &call[..] else { panic!("{call:?}") };
        assert_eq!((call.id.as_str(), call.name.as_str()), ("toolu_1", "mcp__c__read_file"));
        assert_eq!(call.input["path"], "a.rs");

        let end = process_line(r#"{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":10,"output_tokens":5}}}"#);
        let [SubprocessEvent::StepEnd { stop_reason, usage }] = &end[..] else { panic!("{end:?}") };
        assert_eq!(stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(usage.unwrap().input_tokens, 10);
    }

    #[test]
    fn result_and_rate_limit_lines_are_forwarded() {
        let result = process_line(r#"{"type":"result","subtype":"success","is_error":false,"result":"Hi"}"#);
        assert!(matches!(&result[..], [SubprocessEvent::Result(r)] if r.result.as_deref() == Some("Hi")));
        let limit = process_line(r#"{"type":"rate_limit_event","rate_limit_info":{"status":"allowed"}}"#);
        assert!(matches!(&limit[..], [SubprocessEvent::RateLimit(_)]));
    }

    #[test]
    fn garbage_is_ignored() {
        assert!(process_line("").is_empty());
        assert!(process_line("not json").is_empty());
        assert!(process_line(r#"{"type":"user","message":{"content":[{"type":"tool_result"}]}}"#).is_empty());
    }
}
