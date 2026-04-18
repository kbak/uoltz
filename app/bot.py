"""Main bot loop: polls Signal for messages and routes them to the agent.

Uses a message queue so the agent processes one request at a time,
avoiding thread-safety issues with shared conversation state.
Ack messages fire instantly; the agent work is serialized.
"""

import asyncio
import base64
import queue
import re
import sys
import time
import logging
import logging.handlers
import threading
from collections import deque
from pathlib import Path

import httpx

import config
from runtime import state
from signal_client import SignalClient
from agent import (
    create_agent, get_agent, get_agent_for, get_registry, refresh_system_prompt,
    list_available_models, get_current_model_id, get_current_max_tokens,
    server_reload_model,
)
from skills import SkillRegistry
from transcribe import download_and_transcribe, AUDIO_CONTENT_TYPES
import tts as tts_module

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Spoken-digit maps for building voice wake-word patterns
_DIGIT_SPOKEN = {
    "0": r"(?:zero|o+|oh+|nought)",
    "1": r"(?:one)",
    "2": r"(?:two)",
    "3": r"(?:three)",
    "4": r"(?:four)",
    "5": r"(?:five)",
    "6": r"(?:six)",
    "7": r"(?:seven)",
    "8": r"(?:eight)",
    "9": r"(?:nine)",
}
_SEP = r"[\s\-,]*"  # optional separator between spoken digits

def _build_wake_pattern(bot_name: str) -> str:
    """Build a regex that matches the bot name spoken aloud.

    E.g. "006" matches: 006, zero zero six, double-o six, o-o-six, oh oh six, etc.
    Falls back to a literal match for names with no digit map.
    """
    digits = list(bot_name)

    # Handle "double-X" shorthand for repeated leading digits
    # e.g. "006" → also match "double o/oh/zero six"
    spoken_parts = [_DIGIT_SPOKEN.get(d, re.escape(d)) for d in digits]

    # Build the digit-by-digit pattern
    sequential = _SEP.join(spoken_parts)

    # Also allow "double <x> <y>" for a leading pair like "00"
    extras = []
    if len(digits) >= 2 and digits[0] == digits[1]:
        rest_spoken = _SEP.join(spoken_parts[2:]) if len(spoken_parts) > 2 else ""
        double_x = _DIGIT_SPOKEN.get(digits[0], re.escape(digits[0]))
        double_part = rf"double[\s\-]*{double_x}"
        if rest_spoken:
            extras.append(double_part + _SEP + rest_spoken)
        else:
            extras.append(double_part)

    # Literal digits as fallback (e.g. someone types "006" in a voice transcription)
    literal = re.escape(bot_name)

    alts = [sequential, literal] + extras
    return r"(?:hey\s+)?(?:" + "|".join(alts) + r")(?:\s*[,.]?)?"


def _fetch_image_b64(signal_api_url: str, attachment_id: str, content_type: str) -> dict | None:
    """Download an image attachment and return an OpenAI-compatible image content block."""
    try:
        url = f"{signal_api_url.rstrip('/')}/v1/attachments/{attachment_id}"
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        data = base64.standard_b64encode(resp.content).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{data}"},
        }
    except Exception as exc:
        logger.error("Failed to fetch image attachment %s: %s", attachment_id, exc)
        return None
from scheduler import start_scheduler

_LOG_DIR = Path("data/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(level=logging.INFO, format=_LOG_FMT)

# File handler — all modules (including skills) inherit this automatically
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_LOG_FMT))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("signal-bot")

POLL_INTERVAL = 1
ACK_MESSAGE = "⏳ Got it, working on it..."

# Global message queue — the polling loop enqueues, the worker dequeues
_work_queue: queue.Queue = queue.Queue()

# Agent timeout and stop support
AGENT_TIMEOUT = 60  # seconds

# A dedicated event loop running in its own thread so we can create
# cancellable asyncio Tasks for each agent call.
_agent_loop = asyncio.new_event_loop()
threading.Thread(target=_agent_loop.run_forever, daemon=True, name="agent-loop").start()

_current_task: list[asyncio.Task | None] = [None]  # mutable container for cross-scope writes
_current_task_lock = threading.Lock()

# Per-group recent message buffer for topic context
_group_history: dict[str, deque] = {}
_group_history_count: dict[str, int] = {}  # total messages ever appended, per group
_group_last_seen: dict[str, int] = {}      # count at last bot interaction, per group
_GROUP_HISTORY_MAX = 30

# ── Direct skill invocation (bypasses LLM tool selection) ────────────

def handle_direct_skill(cmd: str, args: str, signal: SignalClient, sender: str, images: list | None = None) -> bool:
    """Try to handle a direct skill invocation via registry commands. Returns True if handled."""
    registry = get_registry()
    command = cmd.lower()

    if command not in registry.commands:
        return False

    dc = registry.commands[command]

    if dc.arg_name and not args.strip() and not images:
        usage = dc.usage or f"{command} <input>"
        signal.send(sender, f"Usage: {usage}")
        return True

    # Ack instantly, queue the work
    _work_queue.put(("direct_skill", signal, sender, command, dc, args.strip(), images or []))
    return True


# ── Slash command handler ────────────────────────────────────────────

def handle_slash_command(cmd: str, signal: SignalClient, sender: str) -> bool:
    """Handle a slash command instantly. Returns True if handled."""
    parts = cmd.strip().split(None, 2)
    command = parts[0].lower()
    arg1 = parts[1].lower() if len(parts) > 1 else ""
    arg2 = parts[2] if len(parts) > 2 else ""

    if command == "/stop":
        with _current_task_lock:
            task = _current_task[0]
        if task and not task.done():
            _agent_loop.call_soon_threadsafe(task.cancel)
            signal.send(sender, "🛑 Stopping current query...")
        else:
            signal.send(sender, "Nothing is running.")
        return True

    if command == "/history":
        from agent import _agents
        try:
            n = int(arg1) if arg1 else 10
        except ValueError:
            n = 10
        agent = _agents.get(sender)
        if not agent or not hasattr(agent, "messages") or not agent.messages:
            signal.send(sender, "No agent history for this chat yet.")
            return True
        msgs = agent.messages[-n:]
        lines = [f"🧠 Agent history (last {len(msgs)}/{len(agent.messages)} msgs):\n"]
        for i, m in enumerate(msgs, 1):
            role = m.get("role", "?") if isinstance(m, dict) else getattr(m, "role", "?")
            content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            # content can be a string or a list of blocks
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            parts.append(c.get("text", ""))
                        elif "text" in c:
                            parts.append(str(c["text"]))
                        else:
                            parts.append(f"[{c.get('type', 'block')}]")
                    else:
                        parts.append(str(c))
                content_str = " ".join(parts)
            else:
                content_str = str(content)
            snippet = content_str.replace("\n", " ⏎ ")[:400]
            lines.append(f"[{i}] {role}: {snippet}")
        signal.send(sender, "\n\n".join(lines))
        return True

    if command == "/help":
        registry = get_registry()
        cmd_help = registry.commands_help()
        skill_section = f"Direct skills (bypass LLM routing):\n{cmd_help}\n\n" if cmd_help else ""
        signal.send(sender, (
            f"Available commands:\n\n"
            f"{skill_section}"
            f"Bot controls:\n"
            "  /help  —  Show this message\n"
            "  /stop  —  Cancel the current running query\n"
            "  /history [n]  —  Show last n messages of agent history (default 10)\n"
            "  /model  —  Show current model info\n"
            "  /model list  —  List available models\n"
            "  /model load <name|#>  —  Switch model\n"
            "  /maxlen <n>  —  Set max response length\n"
            "  /context <n>  —  Reload model with new context window\n"
            "  /skills  —  List loaded skills\n"
            "  /schedules  —  List scheduled jobs\n"
            "  /md on|off  —  Toggle markdown formatting\n"
            "  /debug on|off  —  Toggle debug metrics\n"
            "\nAnything without / is sent to the AI agent."
        ))
        return True

    if command == "/model":
        if not arg1:
            signal.send(sender, (
                f"🤖 Model: {get_current_model_id()}\n"
                f"🔗 Server: {config.llm.base_url}\n"
                f"🌡️ Temperature: {config.llm.temperature}\n"
                f"📏 Max tokens: {get_current_max_tokens()}"
            ))
            return True

        if arg1 == "list":
            models = list_available_models()
            if not models:
                signal.send(sender, "Could not fetch models from the server.")
            else:
                current = get_current_model_id()
                lines = []
                for i, m in enumerate(models, 1):
                    marker = " ◀ active" if m == current else ""
                    lines.append(f"  [{i}] {m}{marker}")
                signal.send(sender, f"Available models ({len(models)}):\n\n" + "\n".join(lines))
            return True

        if arg1 == "load" and arg2:
            query = arg2.strip()
            models = list_available_models()
            try:
                idx = int(query)
                if 1 <= idx <= len(models):
                    resolved = models[idx - 1]
                else:
                    signal.send(sender, f"Index {idx} out of range. Use /model list to see 1-{len(models)}.")
                    return True
            except ValueError:
                query_lower = query.lower()
                matches = [m for m in models if query_lower in m.lower()]
                if len(matches) == 1:
                    resolved = matches[0]
                elif len(matches) > 1:
                    lines = [f"Multiple matches for \"{query}\":\n"]
                    for i, m in enumerate(matches, 1):
                        lines.append(f"  [{i}] {m}")
                    lines.append("\nBe more specific or use the number from /model list.")
                    signal.send(sender, "\n".join(lines))
                    return True
                else:
                    signal.send(sender, f"No model matching \"{query}\". Use /model list to see available models.")
                    return True

            signal.send(sender, f"🔄 Loading model: {resolved}...")
            try:
                create_agent(model_id=resolved)
                signal.send(sender, f"✅ Switched to: {resolved}")
            except Exception as e:
                signal.send(sender, f"Failed to load model: {e}")
            return True

    if command == "/skills":
        registry = get_registry()
        if not registry.skills:
            signal.send(sender, "No skills loaded.")
            return True
        lines = [f"Loaded {len(registry.skills)} skill(s), {len(registry.tools)} tool(s):\n"]
        for s in registry.skills:
            tool_names = ", ".join(ref.split(":")[-1] for ref in s.tools)
            lines.append(f"📦 {s.name} v{s.version}\n   {s.description}\n   Tools: {tool_names}\n")
        signal.send(sender, "\n".join(lines))
        return True

    if command == "/md":
        if arg1 == "on":
            state.markdown = True
            refresh_system_prompt()
            signal.send(sender, "✅ Markdown formatting ON")
            return True
        elif arg1 == "off":
            state.markdown = False
            refresh_system_prompt()
            signal.send(sender, "✅ Markdown formatting OFF")
            return True

    if command == "/debug":
        if arg1 == "on":
            state.debug = True
            signal.send(sender, "✅ Debug mode ON — metrics will follow each response")
            return True
        elif arg1 == "off":
            state.debug = False
            signal.send(sender, "✅ Debug mode OFF")
            return True

    if command == "/schedules":
        from scheduler import _load_jobs
        jobs = _load_jobs()
        if not jobs:
            signal.send(sender, "No scheduled jobs found in schedules/")
        else:
            lines = [f"Scheduled jobs ({len(jobs)}):\n"]
            for j in jobs:
                lines.append(f"📅 {j.name}\n   Schedule: {j.schedule}\n   Recipient: {j.recipient}\n   Prompt: {j.prompt[:80]}...\n")
            signal.send(sender, "\n".join(lines))
        return True

    if command == "/maxlen" and arg1:
        try:
            tokens = int(arg1)
            if tokens < 128 or tokens > 1_000_000:
                signal.send(sender, "Value must be between 128 and 1000000.")
                return True
            state.max_tokens = tokens
            model_id = get_current_model_id()
            create_agent(model_id=model_id)
            signal.send(sender, f"✅ Max response length set to {tokens} tokens")
        except ValueError:
            signal.send(sender, "Usage: /maxlen <number>  (e.g. /maxlen 8192)")
        return True

    if command == "/context" and arg1:
        try:
            ctx_size = int(arg1)
            if ctx_size < 512 or ctx_size > 1_000_000:
                signal.send(sender, "Context window must be between 512 and 1000000.")
                return True
            model_id = get_current_model_id()
            signal.send(sender, f"🔄 Reloading {model_id} with context_length={ctx_size}...")
            try:
                result_msg = server_reload_model(model_id, ctx_size)
                create_agent(model_id=model_id)
                signal.send(sender, f"✅ {result_msg}")
            except Exception as e:
                signal.send(sender, f"Failed to reload model: {e}")
        except ValueError:
            signal.send(sender, "Usage: /context <number>  (e.g. /context 16384)")
        return True

    return False

# ── Message handling ─────────────────────────────────────────────────

def _format_debug_info(result) -> str:
    """Extract debug metrics from an AgentResult."""
    try:
        summary = result.metrics.get_summary()
        usage = summary.get("accumulated_usage", {})
        cycles = summary.get("total_cycles", "?")
        duration = summary.get("total_duration", 0)
        tool_usage = summary.get("tool_usage", {})

        lines = [
            "🔍 DEBUG INFO",
            f"  Cycles: {cycles}",
            f"  Duration: {duration:.1f}s",
            f"  Tokens: {usage.get('inputTokens', '?')} in / {usage.get('outputTokens', '?')} out / {usage.get('totalTokens', '?')} total",
        ]
        if tool_usage:
            lines.append("  Tools used:")
            for tool_name, metrics in tool_usage.items():
                count = metrics.get("count", "?")
                avg = metrics.get("average_duration", 0)
                lines.append(f"    - {tool_name}: {count}x, avg {avg:.1f}s")
        return "\n".join(lines)
    except Exception as e:
        return f"🔍 DEBUG: Could not extract metrics: {e}"


def extract_messages(raw: list[dict]) -> list[dict]:
    """Extract messages from Signal API response.

    Returns dicts with: sender, text, attachments, group_id, mentions, timestamp.
    """
    messages = []
    for envelope_wrapper in raw:
        envelope = envelope_wrapper.get("envelope", {})
        sender = envelope.get("source", "")
        data = envelope.get("dataMessage", {})
        text = data.get("message", "") or ""
        attachments = data.get("attachments", []) or []
        mentions = data.get("mentions", []) or []
        timestamp = envelope.get("timestamp") or data.get("timestamp")

        # Group messages have groupInfo with groupId
        group_info = data.get("groupInfo", {})
        group_id = group_info.get("groupId") if group_info else None

        if sender and (text or attachments):
            messages.append({
                "sender": sender,
                "text": text,
                "attachments": attachments,
                "group_id": group_id,
                "mentions": mentions,
                "timestamp": timestamp,
            })
    return messages


# ── Queue worker ─────────────────────────────────────────────────────

def _worker(signal: SignalClient):
    """Single worker thread that processes queued messages sequentially."""
    while True:
        try:
            item = _work_queue.get()
            if item is None:
                break

            msg_type = item[0]

            if msg_type == "agent":
                _, _signal, sender, text, images, voice_reply, orig_sender, orig_timestamp = item
                try:
                    agent = get_agent_for(sender)

                    async def _run_agent():
                        if images:
                            content = [{"type": "text", "text": text}] + images
                            return await agent.invoke_async(content)
                        return await agent.invoke_async(text)

                    async def _run_with_registration():
                        with _current_task_lock:
                            _current_task[0] = asyncio.current_task()
                        try:
                            return await asyncio.wait_for(_run_agent(), timeout=AGENT_TIMEOUT)
                        finally:
                            with _current_task_lock:
                                _current_task[0] = None

                    future = asyncio.run_coroutine_threadsafe(_run_with_registration(), _agent_loop)
                    try:
                        result = future.result()
                        reply = str(result)
                    except asyncio.TimeoutError:
                        logger.warning("Agent timed out for %s after %ds", sender, AGENT_TIMEOUT)
                        reply = f"⏱️ Timed out after {AGENT_TIMEOUT}s."
                    except asyncio.CancelledError:
                        logger.info("Agent cancelled for %s", sender)
                        reply = "🛑 Stopped."
                except Exception as e:
                    logger.exception("Agent error for %s", sender)
                    reply = f"Sorry, I hit an error: {e}"
                    _signal.send(sender, reply)
                    _work_queue.task_done()
                    continue

                _signal.send(sender, reply)
                logger.info("Replied to %s (%d chars)", sender, len(reply))

                if voice_reply and config.tts.enabled:
                    try:
                        _signal.react(sender, orig_sender, orig_timestamp, "🔊")
                        ogg = tts_module.synthesize(reply)
                        _signal.send_voice(sender, ogg)
                        logger.info("Sent voice reply to %s (%d bytes)", sender, len(ogg))
                    except Exception:
                        logger.exception("TTS failed for %s", sender)

                if state.debug:
                    debug_msg = _format_debug_info(result)
                    _signal.send(sender, debug_msg)

            elif msg_type == "direct_skill":
                _, _signal, sender, command, dc, args, images = item

                def _status(msg: str):
                    _signal.send(sender, msg)

                try:
                    if dc.arg_name:
                        kwargs = {dc.arg_name: args}
                        if images:
                            kwargs["images"] = images
                        kwargs["status_fn"] = _status
                        try:
                            result = dc.func(**kwargs)
                        except TypeError:
                            # Skill doesn't accept status_fn or images — call without
                            kwargs.pop("status_fn", None)
                            kwargs.pop("images", None)
                            try:
                                result = dc.func(**{dc.arg_name: args, "images": images})
                            except TypeError:
                                result = dc.func(**{dc.arg_name: args})
                    else:
                        result = dc.func()
                    reply = str(result) if result else "(no output)"
                except Exception as e:
                    logger.exception("Direct skill %s failed", command)
                    reply = f"Error: {e}"

                _signal.send(sender, reply)
                logger.info("Direct skill %s replied to %s (%d chars)", command, sender, len(reply))

            _work_queue.task_done()

        except Exception:
            logger.exception("Worker error")

# ── Main loop ────────────────────────────────────────────────────────

def main():
    cfg_signal = config.signal

    if not cfg_signal.number:
        logger.error("SIGNAL_NUMBER not set. Copy .env.example to .env and configure it.")
        sys.exit(1)

    allowed = cfg_signal.allowed_numbers or None
    signal = SignalClient(cfg_signal.api_url, cfg_signal.number)

    if not signal.is_healthy():
        logger.error(
            "Cannot reach signal-cli-rest-api at %s. "
            "Make sure Docker is running: docker compose up -d signal-api",
            cfg_signal.api_url,
        )
        sys.exit(1)

    logger.info("Signal API is healthy at %s", cfg_signal.api_url)
    logger.info("Creating agent with model %s on %s", config.llm.model_id, config.llm.base_url)

    create_agent()
    registry = get_registry()

    logger.info(
        "Bot is running with %d skill(s), %d tool(s). Polling every %ds...",
        len(registry.skills), len(registry.tools), POLL_INTERVAL,
    )

    # Start the sequential worker thread
    worker_thread = threading.Thread(target=_worker, args=(signal,), daemon=True, name="worker")
    worker_thread.start()

    # Start the proactive scheduler
    start_scheduler(get_agent(), signal)

    while True:
        try:
            raw = signal.receive()
            for msg in extract_messages(raw):
                sender = msg["sender"]
                text = msg["text"]
                attachments = msg["attachments"]
                group_id = msg["group_id"]
                mentions = msg["mentions"]
                timestamp = msg["timestamp"]

                # In groups, reply to the group; in 1:1, reply to the sender
                reply_to = group_id if group_id else sender

                if allowed and sender not in allowed:
                    logger.warning("Ignoring message from unauthorized number: %s", sender)
                    continue

                # Group message filtering: only respond to prefix, mention, voice wake-word, or / commands
                if group_id:
                    # Buffer every group message for context
                    if group_id not in _group_history:
                        _group_history[group_id] = deque(maxlen=_GROUP_HISTORY_MAX)
                        _group_history_count[group_id] = 0
                    _group_history[group_id].append((sender, text.strip()))
                    _group_history_count[group_id] += 1

                    prefix = config.signal.group_prefix.lower()
                    text_lower = text.strip().lower()
                    bot_number = config.signal.number
                    bot_mentioned = any(
                        m.get("number") == bot_number or m.get("uuid") == bot_number
                        for m in mentions
                    )
                    has_audio = any(a.get("contentType", "") in AUDIO_CONTENT_TYPES for a in attachments)

                    if text_lower.startswith(prefix):
                        text = text.strip()[len(prefix):].strip()
                        logger.info("Group %s, from %s (prefix matched): %s", group_id, sender, text[:80])
                    elif bot_mentioned:
                        # Strip leading U+FFFC mention character and whitespace
                        text = re.sub(r"^[\uFFFC\s]+", "", text).strip()
                        logger.info("Group %s, from %s (mention matched): %s", group_id, sender, text[:80])
                    elif text.strip().startswith("/"):
                        logger.info("Group %s, from %s (command): %s", group_id, sender, text[:80])
                    elif has_audio:
                        pass  # defer decision until after transcription — wake-word check below
                    else:
                        continue

                # Transcribe voice messages
                audio_atts = [a for a in attachments if a.get("contentType", "") in AUDIO_CONTENT_TYPES]
                is_voice_message = False
                if audio_atts and not text:
                    att = audio_atts[0]
                    att_id = att.get("id", att.get("filename", ""))
                    logger.info("Voice message from %s, attachment: %s", sender, att_id)
                    signal.react(reply_to, sender, timestamp)
                    signal.react(reply_to, sender, timestamp, "🎤")
                    try:
                        text = download_and_transcribe(cfg_signal.api_url, att_id)
                        is_voice_message = True
                        # Backfill the group history entry (was added with empty text before transcription)
                        if group_id and _group_history.get(group_id):
                            _group_history[group_id][-1] = (sender, text)
                            # count was already incremented at append time, no change needed
                    except Exception as e:
                        logger.exception("Transcription failed")
                        signal.send(reply_to, f"Failed to transcribe voice message: {e}")
                        continue

                    # In groups, require a voice wake word — drop silently if absent
                    if group_id:
                        bot_name = config.signal.bot_name  # e.g. "006"
                        # Build spoken variants: "006" → zero-zero-six, double-o-six, o-o-six, oh-oh-six, etc.
                        wake_pattern = _build_wake_pattern(bot_name)
                        m = re.search(wake_pattern, text, re.IGNORECASE)
                        if m:
                            # Strip the wake word and any surrounding punctuation/whitespace
                            text = (text[:m.start()] + text[m.end():]).strip().lstrip(",. ")
                            logger.info("Group %s, from %s (voice wake-word): %s", group_id, sender, text[:80])
                        else:
                            logger.debug("Group %s: voice message without wake word, ignoring", group_id)
                            continue
                    else:
                        signal.send(reply_to, f'📝 Heard: "{text}"')

                # Fetch image attachments for vision-capable model
                image_atts = [a for a in attachments if a.get("contentType", "") in IMAGE_CONTENT_TYPES]
                images = []
                for att in image_atts:
                    att_id = att.get("id", att.get("filename", ""))
                    ct = att.get("contentType", "image/jpeg")
                    block = _fetch_image_b64(cfg_signal.api_url, att_id, ct)
                    if block:
                        images.append(block)

                if not text and not images:
                    continue

                if not group_id:
                    logger.info("Message from %s: %s", sender, text[:80])

                # If only an image was sent with no text, add a default prompt
                if not text and images:
                    text = "What's in this image?"

                if text.strip().startswith("/"):
                    parts = text.strip().split(None, 1)
                    skill_cmd = parts[0]
                    skill_args = parts[1] if len(parts) > 1 else ""
                    if handle_direct_skill(skill_cmd, skill_args, signal, reply_to, images=images):
                        continue

                    if handle_slash_command(text, signal, reply_to):
                        continue

                # Ack with robot emoji reaction, queue for sequential processing
                signal.react(reply_to, sender, timestamp)

                # Prepend group messages the bot hasn't seen yet (since last interaction)
                if group_id:
                    total = _group_history_count.get(group_id, 0)
                    last_seen = _group_last_seen.get(group_id, 0)
                    unseen_count = total - last_seen - 1  # exclude current message
                    if unseen_count > 0:
                        recent = list(_group_history.get(group_id, []))
                        unseen = recent[-unseen_count - 1:-1]  # messages since last interaction
                        if unseen:
                            history_lines = "\n".join(f"{s}: {t}" for s, t in unseen)
                            text = f"<group_history>\n{history_lines}\n</group_history>\n\n[User asks] {text}"
                    _group_last_seen[group_id] = total

                _work_queue.put(("agent", signal, reply_to, text, images, is_voice_message, sender, timestamp))
                pending = _work_queue.qsize()
                if pending > 1:
                    signal.send(reply_to, f"📋 Queued (position {pending})")

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception:
            logger.exception("Unexpected error in main loop")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
