"""P5B: voice preview synthesis and disk cache.

Implements POST /api/voices/{name}/preview payloads: a fixed sample sentence,
rendered with the voice's own engine, encoded as a mono 24 kHz WAV and cached
at ``<data>/outputs/previews/<sha256(name + "\\0" + engine + "\\0" + sample)>``.wav.
On a cache hit the route reads the file directly and never touches an engine.
"""

from __future__ import annotations

import hashlib
import io
import threading
import wave
from pathlib import Path

import numpy as np

SAMPLE_TEXT = (
    "This is a preview of this voice, reading a short passage "
    "so you can hear how it sounds."
)

# One loaded engine per quality, shared by all preview requests. Engines are
# expensive to construct (model load + canary synthesis), so the first preview
# on a quality pays the load and later requests reuse it; the convert pipeline
# runs engines in its own subprocesses and is unaffected by this cache.
_ENGINES: dict[str, object] = {}
_ENGINES_LOCK = threading.Lock()


def cache_key(name: str, engine: str, sample_text: str = SAMPLE_TEXT) -> str:
    payload = f"{name}\0{engine}\0{sample_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preview_path(cache_dir: str | Path, name: str, engine: str) -> Path:
    return Path(cache_dir) / f"{cache_key(name, engine)}.wav"


def get_engine(quality: str, models_dir: str | Path | None = None) -> object:
    """Return the shared engine for quality, constructing it on first use."""
    with _ENGINES_LOCK:
        engine = _ENGINES.get(quality)
        if engine is None:
            from ..engine import find_models_dir
            from ..engine_pool import make_engine

            engine = make_engine(quality, find_models_dir(models_dir))
            _ENGINES[quality] = engine
        return engine


def synthesize_preview(
    quality: str, voice_config: dict, models_dir: str | Path | None = None
) -> tuple[np.ndarray, int]:
    """Synthesize SAMPLE_TEXT at default speed/lang.

    Returns (float32 mono samples, sample_rate); both real engines emit
    24 kHz sample rates.
    """
    engine = get_engine(quality, models_dir)
    return engine.synthesize(
        SAMPLE_TEXT, voice_config=voice_config, speed=1.0, lang="en-us"
    )


def write_wav_bytes(path: str | Path, samples, sample_rate: int) -> bytes:
    """Encode float32 mono samples as 16-bit PCM WAV, cache at path, return bytes.

    Written atomically (tmp file + rename) so concurrent preview requests
    never expose a half-written cache entry.
    """
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())
    data = buf.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data
