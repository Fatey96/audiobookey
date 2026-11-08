"""Prosody + voice markup parsing.

Page text may contain:
  [pause:N.N]            explicit silence of N.N seconds
  [voice:name]...[/voice]  text synthesized with a named voice (voices.json)
  blank lines (\n\n)     paragraph break -> default paragraph pause

Pause markers are hard boundaries: they split the text, so a synthesized
chunk never contains one. A [voice:] region may not nest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PARAGRAPH_PAUSE = 0.6  # seconds; blank line in the cleaned text
DEFAULT_SENTENCE_GAP = 0.3  # seconds between sentences (--gap)

_PAUSE_RE = re.compile(r"\[pause:([0-9]*\.?[0-9]+)\]")
_VOICE_OPEN_RE = re.compile(r"\[voice:([A-Za-z0-9_.-]+)\]")
_CLOSE_RE = re.compile(r"\[/voice\]", re.IGNORECASE)


class MarkupError(ValueError):
    pass


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    voice: str | None  # voice name from voices.json, or None = page default


@dataclass(frozen=True)
class PauseSegment:
    seconds: float


Segment = SpeechSegment | PauseSegment


class VoiceMap:
    """Ordered regex -> voice-name mapping, applied per chunk of text.

    File format (JSON object, insertion order = priority):

        {"^\\s*[A-Z][A-Z ]+:": "narrator", "\"[^\"]*\"": "dialogue"}

    The first pattern whose regex matches the chunk text decides the voice.
    """

    def __init__(self, patterns: dict[str, str]) -> None:
        import re as _re

        self._rules: list[tuple[_re.Pattern, str]] = []
        for pattern, name in patterns.items():
            try:
                compiled = _re.compile(pattern)
            except _re.error as exc:
                raise MarkupError(
                    f"invalid voice-map regex {pattern!r}: {exc}"
                ) from exc
            self._rules.append((compiled, name))

    @classmethod
    def from_file(cls, path: str | Path) -> "VoiceMap":
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise MarkupError(f"{path}: voice map must be a JSON object")
        return cls(data)

    def match(self, text: str) -> str | None:
        for compiled, name in self._rules:
            if compiled.search(text):
                return name
        return None


# A quoted span is dialogue (straight or typographic quotes).
_QUOTED_SPAN_RE = re.compile(r'[\u201c"][^"\u201d]*[\u201d"]')

# Em-dash (or en-dash) at the start of a line: dialogue line style.
_DIALOGUE_LINE_RE = re.compile(r"(?:^|\n)\s*[\u2014\u2013-]\s")


def quote_units(text: str) -> list[tuple[str, bool]]:
    """Split text at quoted spans into (piece, is_dialogue) pairs.

    The paired quote characters themselves are dropped from the dialogue
    piece (cleaner TTS input); a piece is dialogue iff it comes from a
    quoted span or is an em-dash dialogue line.
    """
    pieces: list[tuple[str, bool]] = []
    cursor = 0
    for match in _QUOTED_SPAN_RE.finditer(text):
        before = text[cursor : match.start()]
        if before.strip():
            pieces.append((before, _DIALOGUE_LINE_RE.search(before) is not None))
        pieces.append((match.group(0)[1:-1], True))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        pieces.append((tail, _DIALOGUE_LINE_RE.search(tail) is not None))
    return pieces


def split_segments(text: str) -> list[Segment]:
    """Split marked-up text into speech and pause segments, in order.

    Speech between markers keeps the voice currently in effect. An unclosed
    [voice:name] region runs to the end of the text (tolerated).
    """
    segments: list[Segment] = []

    def emit_speech(chunk: str, voice: str | None) -> None:
        if chunk.strip():
            segments.append(SpeechSegment(text=chunk.strip(), voice=voice))

    current_voice: str | None = None
    cursor = 0
    while True:
        pause_m = _PAUSE_RE.search(text, cursor)
        voice_m = _VOICE_OPEN_RE.search(text, cursor)
        close_m = _CLOSE_RE.search(text, cursor)

        earliest = min(
            (m.start() for m in (pause_m, voice_m, close_m) if m), default=None
        )
        if earliest is None:
            emit_speech(text[cursor:], current_voice)
            return segments

        if pause_m and pause_m.start() == earliest:
            emit_speech(text[cursor:pause_m.start()], current_voice)
            segments.append(PauseSegment(seconds=float(pause_m.group(1))))
            cursor = pause_m.end()
        elif voice_m and voice_m.start() == earliest:
            if current_voice is not None:
                raise MarkupError(
                    "nested [voice:...] tags are not supported at position "
                    f"{voice_m.start()}: already inside "
                    f"[voice:{current_voice}]"
                )
            emit_speech(text[cursor:voice_m.start()], current_voice)
            current_voice = voice_m.group(1)
            cursor = voice_m.end()
        else:  # close marker
            emit_speech(text[cursor:close_m.start()], current_voice)
            current_voice = None
            cursor = close_m.end()


def validate_segments(segments: list[Segment]) -> None:
    for seg in segments:
        if isinstance(seg, PauseSegment) and seg.seconds < 0:
            raise MarkupError(f"pause cannot be negative: {seg.seconds}")
