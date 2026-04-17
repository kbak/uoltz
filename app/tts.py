"""Text-to-speech using Kokoro (local, GPU-accelerated)."""

import logging
import subprocess
import tempfile
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        cfg = config.tts
        logger.info("Loading Kokoro TTS pipeline (lang=%s, voice=%s)...", cfg.lang, cfg.voice)
        _pipeline = KPipeline(lang_code=cfg.lang)
        logger.info("Kokoro TTS pipeline loaded.")
    return _pipeline


def synthesize(text: str) -> bytes:
    """Synthesize text to OGG/Opus bytes suitable for Signal voice notes."""
    import numpy as np
    import soundfile as sf

    cfg = config.tts
    pipeline = _get_pipeline()

    samples = []
    for _, _, audio in pipeline(text, voice=cfg.voice, speed=cfg.speed):
        if audio is not None:
            samples.append(audio)

    if not samples:
        raise RuntimeError("Kokoro produced no audio")

    audio_np = np.concatenate(samples)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        wav_path = wav_tmp.name

    try:
        sf.write(wav_path, audio_np, samplerate=24000)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_tmp:
            ogg_path = ogg_tmp.name

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", wav_path,
                    "-c:a", "libopus", "-b:a", "24k",
                    "-vbr", "on", "-application", "voip",
                    ogg_path,
                ],
                check=True,
                capture_output=True,
            )
            return Path(ogg_path).read_bytes()
        finally:
            Path(ogg_path).unlink(missing_ok=True)
    finally:
        Path(wav_path).unlink(missing_ok=True)
