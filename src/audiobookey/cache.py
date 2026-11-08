"""Content-hash chunk cache: nothing is ever synthesized twice."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

_SEP = "\x00"


def chunk_key(
    text: str, voice: str, speed: float, engine_version: str, extra: str = ""
) -> str:
    """sha256(text + voice-fingerprint + speed + engine version + extra).

    `voice` is the voice config fingerprint (see voices.voice_fingerprint);
    `extra` carries synthesis parameters that only some engines honor (e.g.
    the sentence gap), so those engines invalidate correctly while others
    keep their cache.
    """
    payload = _SEP.join((text, voice, f"{speed:.6f}", engine_version, extra))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def book_id(pdf_path: str | Path) -> str:
    """Stable per-book id from the PDF's content (renames share the cache)."""
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


class ChunkCache:
    """Filesystem cache of synthesized chunk WAVs, keyed by content hash."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = directory if isinstance(directory, Path) else Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        return self.dir / f"{key}.wav"

    def get(self, key: str) -> Path | None:
        path = self.path_for(key)
        # A WAV header alone (44 bytes) means a previous run died mid-write.
        if path.is_file() and path.stat().st_size > 44:
            self.hits += 1
            return path
        self.misses += 1
        return None

    def store(self, key: str, source: str | Path) -> Path:
        """Move a freshly synthesized wav into the cache atomically."""
        dest = self.path_for(key)
        tmp = self.dir / f".{key}.{os.getpid()}.tmp.wav"
        shutil.move(str(source), str(tmp))
        tmp.replace(dest)
        return dest

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
