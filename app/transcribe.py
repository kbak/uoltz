"""Voice message transcription via the audio-api service."""

import logging

import httpx

import config

logger = logging.getLogger(__name__)

AUDIO_CONTENT_TYPES = {"audio/aac", "audio/mp4", "audio/mpeg", "audio/ogg", "audio/x-m4a"}


def warmup() -> None:
    """Confirm audio-api is reachable and its models are loaded."""
    url = f"{config.audio_api.url.rstrip('/')}/health"
    try:
        resp = httpx.get(url, timeout=10)
        if resp.status_code == 200:
            logger.info("audio-api is ready at %s", config.audio_api.url)
        else:
            logger.warning("audio-api not ready yet (status=%s)", resp.status_code)
    except Exception:
        logger.exception("audio-api health check failed at %s", url)


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file to text via audio-api."""
    url = f"{config.audio_api.url.rstrip('/')}/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        files = {"file": (audio_path.rsplit("/", 1)[-1], f, "application/octet-stream")}
        resp = httpx.post(url, files=files, timeout=120)
    resp.raise_for_status()
    text = resp.json().get("text", "").strip()
    logger.info("Transcribed %s: %d chars", audio_path, len(text))
    return text


def download_and_transcribe(signal_api_url: str, attachment_id: str) -> str:
    """Download an attachment from signal-cli-rest-api and transcribe it via audio-api.

    Streams bytes straight from signal-api into audio-api without touching disk.
    """
    attach_url = f"{signal_api_url.rstrip('/')}/v1/attachments/{attachment_id}"
    transcribe_url = f"{config.audio_api.url.rstrip('/')}/v1/audio/transcriptions"

    with httpx.Client(timeout=60) as client:
        resp = client.get(attach_url)
        resp.raise_for_status()
        audio_bytes = resp.content

        files = {"file": (f"{attachment_id}.m4a", audio_bytes, "audio/mp4")}
        tresp = client.post(transcribe_url, files=files, timeout=120)
        tresp.raise_for_status()
        text = tresp.json().get("text", "").strip()
        logger.info("Transcribed attachment %s: %d chars", attachment_id, len(text))
        return text
