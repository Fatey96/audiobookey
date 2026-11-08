"""Fake TTS engine: deterministic 24 kHz tone, no model, no GPU."""

from __future__ import annotations

import numpy as np


class FakeEngine:
    """Duck-types KokoroTTSEngine: .synthesize and .engine_version."""

    def __init__(self, seconds: float = 0.4, sample_rate: int = 24000) -> None:
        self.seconds = seconds
        self.sample_rate = sample_rate
        self.calls: list[str] = []
        self._version = "fake-engine==1.0;ep=none"

    @property
    def engine_version(self) -> str:
        return self._version

    def synthesize(self, text, voice_config=None, speed=1.0, lang="en-us", gap=None):
        # Different texts give different durations, so we can assert
        # chapter duration ordering in tests.
        self.calls.append(text)
        n = int(self.sample_rate * (self.seconds + len(text) / 5000.0))
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        samples = 0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        return samples, self.sample_rate
