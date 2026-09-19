//! Runs one `claude --print` process per turn and turns its NDJSON output
//! into events.

use crate::types::claude_cli::{ClaudeCliMessage, RateLimitInfo, ResultMessage, StreamEvent};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

const INACTIVITY_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const STDERR_TAIL_LINES: usize = 5;

/// Environment that keeps the CLI a plain model instead of the user's agent.
pub const CLEAN_ENV: [(&str, &str); 4] = [
    // No CLAUDE.md or memory files in the context.
    ("CLAUDE_CODE_DISABLE_CLAUDE_MDS", "1"),
    ("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1"),
    // In -p mode this skips an extra model call that titles every session.
    ("CLAUDE_CODE_DISABLE_TERMINAL_TITLE", "1"),
    // No update, telemetry or feature-flag calls at startup: about 1 s saved per turn.
    ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
];

#[derive(Debug)]
pub enum SubprocessEvent {
    /// The session started; `model` is the id the alias resolved to.
    Init { session_id: String, model: String },
    TextDelta(String),
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
    pub cwd: String,
    pub api: &'static str,
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
    if let Some(session_id) = &options.resume {
        args.extend([
            "--resume".to_string(),
            session_id.clone(),
            "--fork-session".to_string(),
        ]);
    }
    args
}

/// Spawn the CLI, write `input` (one NDJSON user message) to its stdin and
/// forward what it prints. When the receiver goes away, the process is killed.
pub async fn spawn_subprocess(input: String, options: SubprocessOptions, tx: mpsc::Sender<SubprocessEvent>) {
    let rid = options.request_id.clone();
    let start = Instant::now();
    let mode = if options.resume.is_some() { "resume" } else { "fresh" };
    info!("[req={rid}] Spawning claude model={} api={} session={mode}", options.model, options.api);

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
                        warn!("[req={rid}][pid={pid}] Client went away after {:.2}s, killing claude", start.elapsed().as_secs_f64());
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
            () = &mut progress => {
                info!("[req={rid}][pid={pid}] Still running {:.0}s chunks={chunks}", start.elapsed().as_secs_f64());
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
        ClaudeCliMessage::StreamEvent {
            event: StreamEvent::ContentBlockDelta { delta },
        } => match delta.text {
            Some(text) if !text.is_empty() => vec![SubprocessEvent::TextDelta(text)],
            _ => vec![],
        },
        ClaudeCliMessage::RateLimit { rate_limit_info } => vec![SubprocessEvent::RateLimit(rate_limit_info)],
        ClaudeCliMessage::Result(result) => vec![SubprocessEvent::Result(result)],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(resume: Option<&str>) -> SubprocessOptions {
        SubprocessOptions {
            request_id: "r".into(),
            model: "haiku".into(),
            system_prompt: "Be brief.".into(),
            resume: resume.map(String::from),
            cwd: "/tmp".into(),
            api: "openai",
        }
    }

    fn value_after(args: &[String], flag: &str) -> Option<String> {
        args.iter().position(|a| a == flag).map(|i| args[i + 1].clone())
    }

    #[test]
    fn args_run_a_plain_model() {
        let args = build_args(&options(None));
        assert_eq!(value_after(&args, "--tools").as_deref(), Some(""));
        assert_eq!(value_after(&args, "--setting-sources").as_deref(), Some(""));
        assert_eq!(value_after(&args, "--input-format").as_deref(), Some("stream-json"));
        assert_eq!(value_after(&args, "--output-format").as_deref(), Some("stream-json"));
        assert!(args.contains(&"--strict-mcp-config".to_string()));
        assert!(args.contains(&"--disable-slash-commands".to_string()));
        assert!(args.contains(&"--include-partial-messages".to_string()));
    }

    #[test]
    fn args_carry_model_and_system_prompt() {
        let args = build_args(&options(None));
        assert_eq!(value_after(&args, "--model").as_deref(), Some("haiku"));
        assert_eq!(value_after(&args, "--system-prompt").as_deref(), Some("Be brief."));
    }

    #[test]
    fn fresh_sessions_do_not_resume() {
        let args = build_args(&options(None));
        assert!(!args.contains(&"--resume".to_string()));
        assert!(!args.contains(&"--fork-session".to_string()));
        assert!(!args.contains(&"--no-session-persistence".to_string()), "sessions must be saved to be resumed");
    }

    #[test]
    fn resumed_sessions_fork() {
        let args = build_args(&options(Some("abc")));
        assert_eq!(value_after(&args, "--resume").as_deref(), Some("abc"));
        assert!(args.contains(&"--fork-session".to_string()));
    }

    #[test]
    fn clean_env_disables_context_and_background_calls() {
        let names: Vec<&str> = CLEAN_ENV.iter().map(|(k, _)| *k).collect();
        assert!(names.contains(&"CLAUDE_CODE_DISABLE_CLAUDE_MDS"));
        assert!(names.contains(&"CLAUDE_CODE_DISABLE_TERMINAL_TITLE"));
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
    fn assistant_lines_are_not_forwarded() {
        assert!(process_line(r#"{"type":"assistant","message":{"content":[{"type":"text","text":"Hi"}]}}"#).is_empty());
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
        assert!(process_line(r#"{"type":"user"}"#).is_empty());
    }
}
