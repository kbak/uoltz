# Signal AI Agent

A local-first AI chatbot for [Signal](https://signal.org/) messenger with pluggable skills, voice transcription, voice replies, multi-agent orchestration, and proactive scheduled tasks — powered by any OpenAI-compatible LLM server.

## Features

- **100% local** — your LLM, your data, no cloud APIs required
- **Any OpenAI-compatible backend** — LM Studio, Ollama, vLLM, llama.cpp
- **Pluggable skill system** — drop a folder in `app/skills/`, restart, done
- **Voice messages in** — automatic transcription via local Whisper (faster-whisper)
- **Voice messages out** — bot replies with a voice note after every text reply to your voice message (Kokoro TTS)
- **Image understanding** — send a photo and the bot describes or analyzes it
- **Multi-agent brainstorming** — domain-aware parallel ideation with fact-checking, YouTube video analysis, RSS feed context, prior brainstorm awareness, and confidence-scored reports via Strands Agents Graph pattern
- **Real-time research** — web + news search with AI synthesis
- **YouTube summarization** — extract transcripts (captions or Whisper) and produce detailed video summaries
- **URL summarization** — fetch and summarize any web page or article
- **RSS digest** — FreshRSS integration for curated feed summaries
- **Self-extending** — generate new skills from natural language descriptions at runtime
- **Proactive scheduler** — cron-based jobs that send you updates automatically
- **Signal account management** — register numbers, manage groups, all from chat
- **Runtime controls** — switch models, toggle formatting, adjust context window, all via slash commands
- **Per-sender conversation isolation** — each contact has its own agent instance and history
- **Dual deployment** — run on bare metal or in Docker with one command

## Architecture

```
Signal App ↔ signal-cli-rest-api (Docker) ↔ Python Bot ↔ Strands Agent ↔ Local LLM
                                                 │
                                          ┌──────┴──────┐
                                    Skill Registry   Scheduler
                                  (auto-discovered)  (cron jobs)
```

## Quick Start

### Prerequisites

The setup scripts check for these automatically and will install what they can:

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (for host mode)
- Docker or Finch (for container mode and signal-api)
- `ffmpeg` (for voice transcription and TTS encoding)
- `espeak-ng` (for Kokoro TTS phonemization — installed automatically in Docker)
- `curl`
- An OpenAI-compatible LLM server ([LM Studio](https://lmstudio.ai), [Ollama](https://ollama.ai), etc.)

### 1. Clone and configure

```bash
git clone https://github.com/maciejjedrzejczyk/uoltz
cd signal-agent
cp .env.example .env
# Edit .env — set your Signal number and LLM server details
```

### 2. Start the LLM server

Start your local LLM with tool-calling support:

- **LM Studio**: Load a model (e.g. Qwen 2.5 14B), enable "Serve on Local Network", start the server
- **Ollama**: `ollama serve` then `ollama pull qwen2.5:14b`

### 3. Run the bot

**Option A — Host mode** (recommended for development, gives access to local filesystem):

```bash
./scripts/run-host.sh
```

Automatically: checks prerequisites → installs missing tools → stops any Docker bot → sets up Python venv → installs dependencies → launches the bot. Signal-api runs in Docker.

**Option B — Docker mode** (fully containerized):

```bash
./scripts/up.sh
```

Builds and starts both signal-api and the bot as containers.

### 4. Connect Signal

**Link an existing account** (recommended):

1. Open `http://localhost:9922/v1/qrcodelink?device_name=signal-agent`
2. On your phone: Signal → Settings → Linked Devices → scan the QR code

**Register a new number:**

```bash
# May require a captcha — see app/skills/signal_admin/account.py for details
curl -X POST http://localhost:9922/v1/register/+1234567890
curl -X POST http://localhost:9922/v1/register/+1234567890/verify/CODE
```

### 5. Send a message

Text or voice-message the bot's number from Signal. Try:
- "Hello, what can you do?"
- "Search the web for the latest Python release"
- "Brainstorm ideas for a personal productivity app"
- Send a voice note — you'll get a text reply followed by a voice note back
- `/help` for all commands

## Slash Commands

All commands respond instantly without hitting the LLM.

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/model` | Show current model, server, temperature, max tokens |
| `/model list` | List all models on the LLM server (numbered) |
| `/model load <#\|name>` | Switch model by index number, partial name, or full name |
| `/maxlen <n>` | Set max response length in tokens |
| `/context <n>` | Reload model on server with new context window (LM Studio) |
| `/skills` | List all loaded skills and their tools |
| `/schedules` | List all active scheduled jobs |
| `/md on\|off` | Toggle markdown formatting in responses |
| `/debug on\|off` | Show execution metrics (cycles, tokens, duration) after each response |
| `/stop` | Cancel the currently running agent query |

## Built-in Skills

See [SKILLS.md](SKILLS.md) for detailed documentation on each skill, and how to create your own skills.

## Proactive Scheduler

The bot can send you messages on a schedule without you asking. Jobs are defined as YAML files in `schedules/`.

### Creating a scheduled job

Create a file in `schedules/` (e.g. `schedules/my_job.yaml`):

```yaml
name: Morning Weather
schedule: "0 7 * * *"           # cron expression: 7:00 AM daily
recipient: "+1234567890"        # who receives the message
prompt: >
  Research the current weather in Warsaw, Poland.
  Give me a brief forecast for today.
enabled: true
```

### Cron expression examples

| Expression | Meaning |
|-----------|---------|
| `0 7 * * *` | Every day at 7:00 AM |
| `0 7 * * 1-5` | Weekdays at 7:00 AM |
| `*/10 * * * *` | Every 10 minutes |
| `0 9,18 * * *` | Twice daily at 9 AM and 6 PM |
| `0 8 * * 1` | Every Monday at 8:00 AM |

### Included examples

| File | Schedule | What it does |
|------|----------|-------------|
| `morning_weather.yaml` | Daily 7 AM | Weather forecast for Warsaw |
| `amzn_stock.yaml` | Every 10 min | AMZN stock price check |
| `_examples.yaml` | (disabled) | Template showing the format |

Jobs are loaded at bot startup. Use `/schedules` to verify what's active.

## Voice Messages

### Incoming (Speech-to-Text)

Send a voice note to the bot and it will:

1. React with 🎤 to acknowledge
2. 📝 Show you what it heard (transcribed text)
3. Process the transcribed text through the agent
4. Reply in text, then send a voice note back

Transcription runs locally via [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Configure in `.env`:

```env
WHISPER_MODEL=small          # tiny, base, small, medium, large-v3
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8    # int8 recommended for CPU
```

### Outgoing (Text-to-Speech)

After every reply to a voice message, the bot synthesizes a voice note using [Kokoro](https://github.com/hexgrad/kokoro) and sends it back. Kokoro runs entirely locally (~165 MB model) and supports GPU acceleration.

Configure in `.env`:

```env
TTS_ENABLED=true
TTS_VOICE=af_heart           # see voice options below
TTS_LANG=a                   # a=en-US, b=en-GB
TTS_SPEED=1.0
```

Available voices:

| Voice | Gender | Accent |
|-------|--------|--------|
| `af_heart` | Female | American (warm, default) |
| `af_bella` | Female | American |
| `am_adam` | Male | American |
| `am_michael` | Male | American |
| `bf_emma` | Female | British |
| `bm_george` | Male | British |

Set `TTS_ENABLED=false` to disable voice replies without removing the feature.

## Image Understanding

Send a photo to the bot (with or without a caption) and it will describe or analyze it. Requires a vision-capable model loaded on your LLM server (e.g. Qwen2.5-VL, LLaVA).

## Scripts

All scripts include prerequisite checks and will guide you through installing missing tools.

| Script | Description |
|--------|-------------|
| `scripts/run-host.sh` | Run bot on host (checks prereqs, sets up venv, launches) |
| `scripts/run-host.sh --stop` | Switch .env back to Docker mode |
| `scripts/run-docker.sh` | Switch to Docker mode and start containers |
| `scripts/up.sh` | Build + start all Docker services |
| `scripts/down.sh` | Stop all Docker services |
| `scripts/reload.sh` | Rebuild + restart bot container only |
| `scripts/build.sh` | Build bot container image |
| `scripts/logs.sh` | Follow bot logs (`logs.sh signal-api` for Signal API) |

## Project Structure

```
signal-agent/
├── app/                        # Bot source code
│   ├── bot.py                  # Main loop: polls Signal, routes messages
│   ├── agent.py                # Agent factory, per-sender isolation, model management
│   ├── config.py               # Centralized config from .env
│   ├── runtime.py              # Mutable runtime state (toggles)
│   ├── signal_client.py        # Signal REST API client (text + voice send)
│   ├── transcribe.py           # Whisper voice transcription (STT)
│   ├── tts.py                  # Kokoro voice synthesis (TTS)
│   ├── scheduler.py            # Proactive cron-based job scheduler
│   ├── requirements.txt
│   └── skills/                 # Auto-discovered skill plugins
│       ├── registry.py         # Skill discovery engine
│       ├── _template/          # Starter template for new skills
│       ├── brainstorm/         # Multi-agent brainstorming (Strands Graph)
│       ├── notes/              # Local note-taking
│       ├── research/           # Real-time web research
│       ├── rss_digest/         # FreshRSS feed digest
│       ├── shell/              # Guarded shell commands
│       ├── skill_builder/      # Runtime skill generation
│       ├── summarize/          # URL/text summarization
│       ├── web_search/         # DuckDuckGo search
│       └── youtube_summary/    # YouTube video summarization
├── schedules/                  # Cron job definitions (YAML)
├── data/                       # Persistent data (gitignored)
├── scripts/                    # Build/run/deploy toolkit
├── docker-compose.yml
├── Dockerfile
├── .env.example                # Configuration template
└── README.md
```

## Configuration

All configuration is in `.env` (created from `.env.example`). The file supports two modes — the launcher scripts toggle between them automatically:

```env
# Docker mode (bot in container)
LLM_BASE_URL=http://host.docker.internal:1234/v1
SIGNAL_API_URL=http://signal-api:8080

# Host mode (bot on bare metal)
LLM_BASE_URL=http://localhost:1234/v1
SIGNAL_API_URL=http://localhost:9922
```

## LLM Server Compatibility

Any server exposing an OpenAI-compatible `/v1/chat/completions` endpoint:

| Server | Base URL | API Key | Notes |
|--------|----------|---------|-------|
| LM Studio | `http://localhost:1234/v1` | `lm-studio` | Enable "Serve on Local Network" for Docker mode |
| Ollama | `http://localhost:11434/v1` | `ollama` | Pull a model with tool support |
| vLLM | `http://localhost:8000/v1` | `token-abc123` | |
| llama.cpp | `http://localhost:8080/v1` | `sk-no-key-required` | |

Pick a model with tool/function-calling support for best results (e.g. Qwen 2.5, Llama 3.1, Mistral). For image understanding, use a vision-capable model (e.g. Qwen2.5-VL).

## Tech Stack

- [Strands Agents](https://strandsagents.com/) — agent framework with multi-agent patterns
- [signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) — Signal messenger REST API
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — local speech-to-text via CTranslate2
- [Kokoro](https://github.com/hexgrad/kokoro) — local text-to-speech (~165 MB, GPU-accelerated)
- [DuckDuckGo Search](https://github.com/deedy5/ddgs) — web search without API keys
- [croniter](https://github.com/kiorky/croniter) — cron expression parsing for the scheduler

## Patches (kbak fork)

This fork applies the following changes on top of upstream:

### `signal_client.py`
- **`receive()`** — replaced retry-based polling with a direct `?timeout=1` long-poll to avoid lock contention with signal-cli
- **`_resolve_group_id()`** — resolves internal group IDs (received in messages) to the `group.XXX=` format required by `/v2/send`
- **`react()`** — sends an emoji reaction to a specific message; used to ack incoming messages with 🤖 instead of a text reply
- **`send()`** — uses `_resolve_group_id()` so group replies are routed correctly
- **`send_voice()`** — sends an OGG/Opus audio file as a Signal voice note via `base64_attachments`

### `bot.py`
- **Poll interval** — reduced from 2s to 1s to match the receive timeout
- **Mention detection** — bot can be triggered in groups via `@mention` (Signal sends these as U+FFFC) in addition to the text prefix
- **Reaction ack** — replaced the "⏳ Got it..." text ack with a 🤖 emoji reaction on the triggering message
- **Per-group history buffer** — last 10 messages in each group are buffered; when the bot is invoked, recent chat is prepended as context so it can handle ambiguous references (e.g. "what do you think about this?")
- **Image support** — image attachments are downloaded and forwarded to the LLM as base64 vision content blocks
- **Voice reply flag** — tracks whether a message originated as a voice note; triggers TTS synthesis after the text reply
- **Skill attribution removed** — tool usage messages are not sent to the user
- **Per-sender agent routing** — uses `get_agent_for(sender)` so each sender has an isolated conversation history

### `agent.py`
- **Per-sender agent isolation** — replaced single global `_agent` with a `dict[str, Agent]` keyed by sender; agents are lazily created and share the same model/registry
- **Conversation history trimming** — each agent's history is capped at 50 messages to prevent unbounded memory growth
- **Language instruction** — system prompt instructs the model to always respond in the user's language
- **Thinking mode disabled** — `/no_think` prepended to system prompt (for Qwen3 models)
- **Prompt injection guard** — system prompt instructs the model to ignore directives found inside `<group_history>` context tags
- **`get_agent_for(sender)`** — retrieves or lazily creates a per-sender agent instance
- **`server_reload_model()`** — reloads the model on LM Studio with a new context window length via the management API

### `tts.py` (new)
- Lazy-loaded Kokoro TTS pipeline singleton
- `synthesize(text) -> bytes` — generates OGG/Opus audio from text via Kokoro → ffmpeg encode at 24 kbps voip profile

### `config.py`
- **`TTSConfig`** — `TTS_ENABLED`, `TTS_VOICE`, `TTS_LANG`, `TTS_SPEED` environment variables with defaults

### `skills/web_search/skill.yaml`
- **Disabled** — replaced by a SearXNG-based custom skill in the deployment stack

## License

MIT
