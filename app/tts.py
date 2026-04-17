"""Text-to-speech using Kokoro ONNX (local, CPU/GPU)."""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_kokoro = None

_MODEL_DIR = Path(os.getenv("TTS_MODEL_DIR", "/app/kokoro-models"))
_VOICES_FILE = _MODEL_DIR / "voices-v1.0.bin"
_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "300"))


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        available = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in available
        provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
        # fp16 is 14x faster on GPU; int8 is better for CPU
        model_file = _MODEL_DIR / ("kokoro-v1.0.fp16.onnx" if use_cuda else "kokoro-v1.0.int8.onnx")
        os.environ["ONNX_PROVIDER"] = provider
        logger.info("Loading Kokoro ONNX model %s (provider=%s)...", model_file.name, provider)
        _kokoro = Kokoro(str(model_file), str(_VOICES_FILE))
        logger.info("Kokoro ONNX model loaded.")
    return _kokoro


def synthesize(text: str) -> bytes:
    """Synthesize text to OGG/Opus bytes suitable for Signal voice notes."""
    import soundfile as sf

    cfg = config.tts
    kokoro = _get_kokoro()

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS].rsplit(" ", 1)[0] + "…"

    samples, sample_rate = kokoro.create(
        text,
        voice=cfg.voice,
        speed=cfg.speed,
        lang="en-us" if cfg.lang == "a" else "en-gb",
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        wav_path = wav_tmp.name

    try:
        sf.write(wav_path, samples, sample_rate)

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
