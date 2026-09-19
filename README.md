# claude-max-api-proxy-rs

English · [Русский](README.ru.md)

A local server with the OpenAI and Anthropic HTTP APIs in front of the `claude` CLI. A client sends an ordinary `/v1/chat/completions` request, the proxy runs `claude --print` with it and returns the answer in the same format.

Setting up JetBrains Air with an agent: [connect the proxy to JetBrains Air through Goose](docs/jetbrains-air.md).

> [!WARNING]
> Requests are billed to the Claude plan the CLI is logged in with. Anthropic [does not allow](https://code.claude.com/docs/en/agent-sdk/overview) third-party products to use claude.ai login or subscription rate limits without its approval, so running other clients on a Pro or Max plan through this proxy puts the account at risk.

## Quick start

You need the Claude Code CLI logged in to your account and a Rust toolchain from [rustup.rs](https://rustup.rs).

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude auth login

git clone https://github.com/GlebSmolyakov/claude-max-api-proxy-rs.git
cd claude-max-api-proxy-rs
cargo install --path .
claude-max-api
```

The server listens on `127.0.0.1:8080`, localhost only. A first request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Name the capital of Portugal in one word."}]}'
```

```json
{
  "id": "chatcmpl-adf63445",
  "object": "chat.completion",
  "created": 1789809523,
  "model": "claude-sonnet-5",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Lisbon" },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 477,
    "completion_tokens": 7,
    "total_tokens": 484,
    "prompt_tokens_details": { "cached_tokens": 0 }
  }
}
```

## Connecting a client

OpenAI-compatible clients take `http://127.0.0.1:8080/v1` as the base URL, Anthropic SDKs take `http://127.0.0.1:8080`. The proxy does not check API keys, so any non-empty string works.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Suggest a name for a function that retries failed HTTP calls."}],
)
print(reply.choices[0].message.content)
```

```python
import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="local")
message = client.messages.create(
    model="sonnet",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Suggest a name for a function that retries failed HTTP calls."}],
)
print(message.content[0].text)
```

Agents work the same way. [Goose](https://github.com/block/goose), for example, reads the address from `ANTHROPIC_HOST`, not from `ANTHROPIC_BASE_URL`:

```bash
export GOOSE_PROVIDER=anthropic GOOSE_MODEL=sonnet
export ANTHROPIC_HOST=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local
goose session
```

As an ACP agent inside an editor, it runs as `goose acp` with the same variables. Step by step for JetBrains Air: [connect the proxy to JetBrains Air through Goose](docs/jetbrains-air.md).

## Endpoints

| Endpoint | Method | What it returns |
|----------|--------|-----------------|
| `/v1/chat/completions` | POST | OpenAI Chat Completions, whole or streamed; `stream_options.include_usage` adds a usage chunk |
| `/v1/messages` | POST | Anthropic Messages, whole or streamed |
| `/v1/models` | GET | The aliases and every model id a request has run on, with its context window once known |
| `/health` | GET | Uptime, CLI version, the model each alias resolved to, turns waiting for tool results, subscription usage |

Responses carry the real model id, token counts including cache reads and writes, and the stop reason; on the OpenAI side `max_tokens` becomes `finish_reason: "length"` and `tool_use` becomes `"tool_calls"`. Errors come in the error format of the endpoint that was called, with a status that says what happened:

| Status | When |
|--------|------|
| 400 | The request is malformed, the model name is unknown, or the API rejected the request, for example when it could not download an image URL |
| 429 | The plan's usage limit is used up |
| 502 | The CLI failed or exited without an answer; the message ends with its last stderr lines |
| 504 | The CLI printed nothing for 30 minutes |

A streamed response holds its headers until the first token, for up to 10 seconds. An error in that window arrives as an HTTP status, not as an event inside a 200 stream.

## Models

| `model` in the request | Passed to `claude --model` |
|------------------------|----------------------------|
| `fable`, `opus`, `sonnet`, `haiku` | As is; the CLI picks the newest model of that family |
| A full id: `claude-sonnet-5`, `claude-opus-5[1m]` | As is |
| `claude-code-cli/<name>` | `<name>`, by the same rules |
| Missing | `opus` |
| Anything else, e.g. `gpt-4o` | Nothing: the proxy answers `400 Unknown model` |

## What the model receives

Each request runs the CLI as a bare model: no built-in tools, MCP servers, skills, settings files, `CLAUDE.md` or memory from the machine, and no background calls at startup. A short request costs under 500 input tokens, and the first token of a short `haiku` reply arrives in about 1.5 seconds.

The system prompt from the request becomes the CLI's system prompt: OpenAI `system` and `developer` messages, Anthropic `system`. Images reach the model as images: `data:` and `http(s)` URLs in OpenAI `image_url` parts, `base64` and `url` sources in Anthropic `image` blocks. The API downloads URL images itself.

`max_tokens`, `temperature`, `stop` and other sampling fields have nowhere to go: the CLI sets them itself.

The CLI also puts a short fixed preamble in front of every subscription request. Without a system prompt of your own, the model may describe itself as running on the Claude Agent SDK.

## Tools

A request can declare tools that the client runs itself: OpenAI `tools` with `function` entries, or Anthropic `tools` with an `input_schema`. The proxy serves them to the CLI as an MCP server at `/mcp/<token>`, so the model calls them like its own. When it does, the response carries the calls (`tool_calls` with `finish_reason: "tool_calls"`, or `tool_use` blocks with `stop_reason: "tool_use"`) and the CLI process waits. The client runs the tools and sends the results the usual way, as `tool` messages or `tool_result` blocks. The proxy hands them to the waiting process, which goes on without a restart: a four-step agent task, such as read a file, edit it, show it and answer, runs on one CLI process.

A process waits up to 30 minutes for tool results, then the proxy stops it. `tool_choice: "none"` (OpenAI) or `{"type": "none"}` (Anthropic) hides the tools; other `tool_choice` values are accepted but not enforced. Anthropic server tools such as web search are skipped.

## Conversations

Clients resend the whole conversation with every request. The proxy keeps track of which CLI session holds each history, so a follow-up resumes that session with `claude --resume <id> --fork-session` and sends only the new message. The model gets a real multi-turn dialogue, and the API can serve the earlier turns from its prompt cache.

Regenerating an earlier reply forks a separate branch that does not mix with the main one. If the proxy has not seen a history, because it was edited or went unused for over 24 hours, it starts a fresh session and passes the earlier turns in as a transcript, images included. The turn after that resumes normally.

The map is stored in `~/.claude-max-api/sessions.json` and survives restarts. Every hour the proxy drops entries unused for 24 hours, together with their CLI transcripts.

## Health and subscription usage

`/health` after a few requests:

```json
{
  "status": "ok",
  "uptime": 53,
  "cli_version": "2.1.276 (Claude Code)",
  "workdir": "/Users/you/.claude-max-api/workdir",
  "saved_sessions": 9,
  "waiting_for_tools": 0,
  "models": { "haiku": "claude-haiku-4-5-20251001" },
  "rate_limits": {
    "status": "allowed",
    "windows": {
      "five_hour": { "utilization": 0.25, "resets_at": 1789824000 },
      "seven_day": { "utilization": 0.03, "resets_at": 1790334000 }
    },
    "reported_at": 1789808589
  }
}
```

`rate_limits` repeats what the CLI reported after the latest request: `utilization` is the used share of each window, `resets_at` a Unix timestamp. Before the first request the field is `null`.

## Options

```bash
claude-max-api [PORT] [--cwd DIR]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `PORT` | `8080` | Port on `127.0.0.1` |
| `--cwd DIR` | `~/.claude-max-api/workdir` | Working directory of the CLI processes; the CLI saves their transcripts under `~/.claude/projects/`, in a folder named after this path |
| `RUST_LOG` | `claude_max_api=info` | Log filter in [tracing `EnvFilter`](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html) syntax; `claude_max_api=debug` also prints the CLI's stderr |

## Development

```bash
cargo test
cargo clippy --all-targets
```

```
src/
├── main.rs          startup, state directory, shutdown
├── server.rs        router and shared state
├── routes.rs        HTTP handlers and streaming
├── conversation.rs  request turns and tools, CLI input, history keys
├── turn.rs          one request: resume, start fresh or continue a parked turn
├── bridge.rs        MCP server that lends client tools to the CLI
├── subprocess.rs    the claude process and its NDJSON output
├── session.rs       history to CLI session map
├── models.rs        accepted model names
├── status.rs        uptime, limits and model ids for /health
├── error.rs         errors in OpenAI and Anthropic shapes
├── types/           OpenAI, Anthropic and CLI message types
└── adapter/         requests in, responses and stream events out
```

## License

MIT, see [LICENSE](LICENSE).
