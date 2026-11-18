# ollama-antigravity

Zero-dependency Go HTTP bridge exposing an Ollama-compatible API backed by a worker pool of local `agy acp` subprocesses over JSON-RPC 2.0 `stdio`.

## Features & Guarantees

- **Go Standard Library Only**: Zero external `require` dependencies in `go.mod`.
- **Zero Direct OAuth / Upstream HTTP**: All inference executes strictly via local `agy acp` child processes over `stdin`/`stdout`.
- **Zero Zombie Processes**: Each worker runs in a dedicated OS-locked thread (`runtime.LockOSThread()`) with `SysProcAttr{Pdeathsig: syscall.SIGKILL, Setpgid: true}` and kills/reaps on timeout or error.
- **Native Antigravity Tool Restriction**: Runs in `HEADLESS INFERENCE MODE` with `--sandbox` (`ACP_SANDBOX=true`) and stream-level neutralization (`ACP_RESTRICT_NATIVE_TOOLS=true`), killing and respawning any worker that attempts native Antigravity tool invocations (`tool_call`/`tool_call_update`). Client-supplied `tools` use the `<tool_call>{...}</tool_call>` protocol.
- **Model Filtering**: Dynamically discovers models from `agy acp` and excludes Orchestrator models (`gemini-pro-orchestrator-g1-pro` and `orchestrator` substring by default via `ACP_EXCLUDE_MODELS`) from `/api/tags`, `/api/show`, `/api/chat`, and `/api/generate`.
- **Ollama & OpenAI `/v1/` Endpoints**:
  - `GET /`, `HEAD /` (`"Ollama is running"`)
  - `GET /api/tags`, `POST /api/show`, `POST /api/chat`, `POST /api/generate`, `GET /api/version`, `GET /api/ps`, `GET /healthz`
  - `GET /v1/`, `GET /v1/models`, `GET /v1/models/{model}`, `POST /v1/chat/completions` (SSE streaming & non-streaming, `tools`, `response_format`), `POST /v1/completions`

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `LISTEN_ADDR` | `0.0.0.0:11435` | HTTP bind address |
| `AGY_BIN` | `agy` | Path or binary name for Antigravity CLI |
| `ACP_WORKERS` | `2` | Number of concurrent `agy acp` workers |
| `ACP_TIMEOUT_SEC` | `120` | Hard per-request timeout before killing and respawning worker |
| `ACP_EMPTY_CWD` | `/tmp/hindsight-acp-empty` | Isolated empty working directory passed to `session/new` |
| `ACP_EXCLUDE_MODELS` | `gemini-pro-orchestrator-g1-pro,orchestrator` | Comma-separated model IDs or substrings to exclude |
| `ACP_DEFAULT_MODEL` | `claude-sonnet-5-5-medium` | Fallback model when request omits `model` |
| `ACP_DISABLE_TELEMETRY` | `true` | Pass `--disable-telemetry` to `agy acp` |
| `ACP_INCLUDE_DEFAULT_CUSTOMIZATIONS` | `false` | Pass `--include-default-customizations=false` when `false` |
| `ACP_SANDBOX` | `true` | Pass `--sandbox` to `agy acp` |
| `ACP_RESTRICT_NATIVE_TOOLS` | `true` | Kill & respawn worker if Antigravity invokes native `tool_call` |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warn`, or `error` |

## Build & Run

```bash
go build -o ollama-antigravity ./cmd/ollama-antigravity
./ollama-antigravity
```

## Arch Linux / systemd

Install via AUR (`yay -S ollama-antigravity`), or build locally from `packaging/aur`.

If running the system unit `ollama-antigravity.service` using an existing user's `agy` session, add a systemd drop-in:

```ini
[Service]
User=<username>
Group=<username>
Environment=HOME=/home/<username>
```

## Hindsight Configuration

```env
HINDSIGHT_API_LLM_PROVIDER=ollama
HINDSIGHT_API_LLM_BASE_URL=http://10.0.0.254:11435
HINDSIGHT_API_LLM_MODEL=claude-sonnet-5-5-medium
```
