"""Text-to-speech via the audio-api service (Kokoro under the hood)."""

import logging

import httpx

import config

logger = logging.getLogger(__name__)


def warmup() -> None:
    """No-op: audio-api loads its own models at startup. Left in place
    so bot.py's existing warmup call continues to work."""
    url = f"{config.audio_api.url.rstrip('/')}/health"
    try:
        resp = httpx.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning("audio-api TTS not ready (status=%s)", resp.status_code)
    except Exception:
        logger.exception("audio-api health check failed at %s", url)


def synthesize(text: str) -> bytes:
    """Synthesize text to OGG/Opus bytes suitable for Signal voice notes."""
    cfg = config.tts
    url = f"{config.audio_api.url.rstrip('/')}/v1/audio/speech"
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": cfg.voice,
        "lang": cfg.lang,
        "speed": cfg.speed,
        "response_format": "ogg",
    }
    resp = httpx.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.content
