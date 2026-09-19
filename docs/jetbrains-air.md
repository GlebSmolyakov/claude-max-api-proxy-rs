# Connect the proxy to JetBrains Air through Goose

English · [Русский](jetbrains-air.ru.md)

JetBrains Air has no setting for a custom API address. Agents in Air are programs: Air starts them itself and talks to them over the Agent Client Protocol (ACP). An agent that speaks ACP and can call the Anthropic API at a given address therefore has to sit between Air and the proxy. [Goose](https://github.com/block/goose) fills that role.

Once set up, Air gets an agent that works in the open project: it reads and edits files and runs commands. The model runs through the `claude` CLI, on the Claude plan the CLI is logged in with.

> [!WARNING]
> Anthropic [does not allow](https://code.claude.com/docs/en/agent-sdk/overview) third-party products to use claude.ai login or subscription rate limits without its approval. A third-party agent running on a Pro or Max plan through this proxy puts the account at risk.

## How the pieces connect

```
Air ──ACP──▶ goose acp ──HTTP──▶ proxy :8080 ──▶ claude CLI ──▶ Claude plan
                 ▲                     │
                 └──── tool calls: read, edit, run
```

Air hands the task to Goose. Goose sends a request to the proxy as if it were the Anthropic API and includes the list of its tools. The proxy starts `claude` and gives it those tools through an MCP server built into the proxy.

When the model asks for a tool, the proxy returns the call to Goose while the `claude` process waits for the result. Goose runs the tool in the project folder and sends the result with its next request. The proxy passes it to the waiting process, and the model carries on. The whole task, from the first step to the answer, runs in one `claude` process.

## Install the components

1. Install the Claude Code CLI and log in with your plan:

   ```bash
   curl -fsSL https://claude.ai/install.sh | bash
   claude auth login
   ```

2. Install Goose:

   ```bash
   brew install block-goose-cli
   ```

3. Build and install the proxy as described in the [README quick start](../README.md#quick-start). Building needs Rust from [rustup.rs](https://rustup.rs).

## Start the proxy

The proxy has to run for as long as the agent is used in Air. Start it in a separate terminal window:

```bash
claude-max-api
```

If the proxy was built with `cargo build --release` rather than installed with `cargo install`, the binary is in the repository folder: `target/release/claude-max-api`.

The proxy listens on `127.0.0.1:8080`. A request to `/health` shows that it is up:

```bash
curl http://127.0.0.1:8080/health
```

The response contains `"status":"ok"`.

## Add the agent to Air

Open *task toolbar > agent selector > Add ACP Agent*. Air opens `acp.json`, which on macOS is `~/Library/Application Support/JetBrains/Air/acp.json`. Replace its contents and save the file:

```json
{
  "agent_servers": {
    "Claude Code Proxy": {
      "command": "/opt/homebrew/bin/goose",
      "args": ["acp"],
      "env": {
        "GOOSE_PROVIDER": "anthropic",
        "GOOSE_MODEL": "sonnet",
        "ANTHROPIC_HOST": "http://127.0.0.1:8080",
        "ANTHROPIC_API_KEY": "sk-local-proxy-key"
      }
    }
  }
}
```

After saving, the `Claude Code Proxy` agent appears in Air's agent list.

| Field | Value | Purpose |
|---|---|---|
| `command` | Path to `goose` | `which goose` prints the full path. A Homebrew install on Apple Silicon puts it at `/opt/homebrew/bin/goose` |
| `args` | `["acp"]` | Runs Goose as an ACP agent that talks to Air over stdin and stdout. `goose serve` starts a network server instead, and Air never gets an answer from it |
| `GOOSE_PROVIDER` | `anthropic` | Goose talks to the proxy using the Anthropic protocol |
| `GOOSE_MODEL` | `sonnet` | The Claude model: `sonnet`, `opus`, `haiku` or `fable`. `opus` and `fable` use up the plan's limit faster |
| `ANTHROPIC_HOST` | `http://127.0.0.1:8080` | The proxy address. Goose does not read `ANTHROPIC_BASE_URL` |
| `ANTHROPIC_API_KEY` | Any string | The proxy does not check keys |
| `GOOSE_MODE` | `approve` or `auto` | Optional. With `approve`, Goose asks before it acts; with `auto`, it acts right away |

## Work with the agent

Pick `Claude Code Proxy` in the agent list and give it a task in the open project. Goose runs commands in the project folder that Air passed to it.

A specific request finishes sooner than a vague one. For "fix the typo in `src/main.rs`", the model opens that file right away. For "find the bug", it may start a search across the whole disk, and such a command runs for minutes.

## Check the proxy log

Every task leaves lines like these in the proxy log:

```
Anthropic messages model=sonnet stream=true turns=1 tools=18
Spawning claude model=sonnet api=anthropic session=fresh tools=on
Waiting for the client to run 1 tool call(s)
Continuing a parked turn with 1 tool result(s)
```

| Line | Meaning |
|---|---|
| `tools=18` | Goose sent its tools. The count depends on the Goose version and enabled extensions |
| `Spawning claude … tools=on` | The proxy started `claude` with the Goose tools |
| `Waiting for the client to run …` | The model called a tool; the `claude` process waits for Goose's result |
| `Continuing a parked turn …` | Goose sent the result; the same `claude` process goes on |

The `rate_limits` field of `/health` shows plan usage: the share of the five-hour and weekly limits already used, and when each window resets.

## Fix common problems

| Symptom | Cause | Fix |
|---|---|---|
| No answer in Air, the loading indicator keeps spinning | `args` says `serve` | Change it to `["acp"]` |
| No new lines in the proxy log | The proxy is not running, or `env` sets `ANTHROPIC_BASE_URL` | Start the proxy and put the address in `ANTHROPIC_HOST` |
| The log says `Ignoring client tools` and the agent does not change files | The running proxy build has no tool support | Update and rebuild the proxy |
| The agent returns an `Unknown model` error | `GOOSE_MODEL` names a model the proxy does not accept | Use `sonnet`, `opus`, `haiku`, `fable` or a full Claude model id |
| Error with status 429 | The plan's usage limit is used up | Wait for the window to reset; `/health` shows when |

## Update the proxy

Pull the changes and rebuild:

```bash
cd claude-max-api-proxy-rs
git pull
cargo install --path .
```

Restart the proxy after the rebuild. Air does not need a restart.

## Limitations

At every step of a task, Goose sends the whole conversation and the descriptions of its tools. Agent work therefore uses up the plan's limit noticeably faster than a chat.

The `claude` process waits up to 30 minutes for a tool result. If Goose sends nothing in that time, for example because the task was cancelled, the proxy stops the process.

Sampling parameters from the request, such as `temperature` and `max_tokens`, do not reach the model: the CLI sets them itself.
