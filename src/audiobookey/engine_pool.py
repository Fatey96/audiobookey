"""Engine creation for the pipeline: draft (Kokoro) + final (Chatterbox)."""

from __future__ import annotations

from .engine import KokoroTTSEngine
from .engine_chatterbox import ChatterboxEngine


def make_engine(quality: str, models_dir) -> object:
    """Create the engine for the requested quality.

    Duck types KokoroTTSEngine + ChatterboxEngine:
        .synthesize(text, voice=..., speed=..., lang=..., gap=...)
        .sample_rate
        .engine_version
    """
    if quality == "draft":
        return KokoroTTSEngine(models_dir)
    if quality == "final":
        return ChatterboxEngine(models_dir)
    raise ValueError(f"unknown engine quality: {quality!r}")
