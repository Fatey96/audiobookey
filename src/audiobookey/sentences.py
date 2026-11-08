"""Sentence-level manifest for converted books.

The manifest maps every spoken sentence to its absolute time range in the
final m4b, so the reader UI can highlight along playback. It is written as
`manifest.json` next to the m4b after encoding:

    {"engine": "kokoro", "speed": 1.0, "sentences": [
        {"page": 1, "index": 0, "start_s": 0.0, "end_s": 2.31,
         "text": "...", "voice": "af_heart"},
        ...
    ]}

Timings reuse the per-chunk WAV durations already measured during assembly;
sentence times inside a chunk are allocated proportionally to sentence
length. A manifest rebuild alone is allowed on a cached re-run (seconds).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .chunk import split_sentences


@dataclass
class TimelineChunk:
    """One synthesized chunk: (page number, chunk text, voice, duration)."""

    page: int
    text: str
    voice: str
    duration_s: float


def build_sentence_entries(timeline: list[TimelineChunk]) -> list[dict]:
    """Walk assembled chunks in play order and emit timed sentence entries."""
    sentences: list[dict] = []
    cursor = 0.0
    index_per_page: dict[int, int] = {}
    for tc in timeline:
        start = cursor
        parts = split_sentences(tc.text) or [tc.text]
        total_chars = sum(len(s) for s in parts) or 1
        for sentence in parts:
            share = tc.duration_s * len(sentence) / total_chars
            sentences.append(
                {
                    "page": tc.page,
                    "index": index_per_page.get(tc.page, 0),
                    "start_s": round(start, 3),
                    "end_s": round(start + share, 3),
                    "text": sentence,
                    "voice": tc.voice,
                }
            )
            index_per_page[tc.page] = index_per_page.get(tc.page, 0) + 1
            start += share
        cursor += tc.duration_s
    return sentences


def write_manifest(
    out_path: str | Path,
    engine: str,
    speed: float,
    sentences: list[dict],
) -> Path:
    """Write manifest.json next to the m4b and return its path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"engine": engine, "speed": speed, "sentences": sentences}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
