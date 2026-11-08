"""Sentence-aware chunking with a phoneme budget for Kokoro (~500 token limit)."""

from __future__ import annotations

import re

from .markup import _QUOTED_SPAN_RE

# Kokoro's hard internal limit is ~510 phoneme tokens; stay under it.
MAX_PHONEMES = 450

# Rough English grapheme->phoneme ratio used when the espeak tokenizer is
# unavailable (fallback only; real counting is preferred).
_FALLBACK_CHARS_PER_PHONEME = 1.25

_SENTENCE_END = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*\s+")
_CLAUSE_BREAK = re.compile(r"(?<=[,;:\u2014\u2013])\s+")

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "fig", "no", "approx", "dept", "est", "inc", "ltd", "co",
}

_WORD_TAIL = re.compile(r"[A-Za-z.]+$")

def split_sentences(text: str) -> list[str]:
    """Split into sentences, tolerating common abbreviations and quotes."""
    text = text.strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        prefix = text[start:match.start()]
        tail = _WORD_TAIL.search(prefix)
        if tail:
            word = tail.group(0).rstrip(".").lower()
            if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue  # initials and abbreviations: not a sentence end
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


class PhonemeCounter:
    """Counts phonemes with espeak when available, else estimates length."""

    def __init__(self) -> None:
        self._tokenizer = None
        try:  # pragma: no cover - exercised only when espeak is importable
            from kokoro_onnx.tokenizer import Tokenizer

            self._tokenizer = Tokenizer()
        except Exception:
            self._tokenizer = None

    @property
    def exact(self) -> bool:
        return self._tokenizer is not None

    def __call__(self, text: str) -> int:
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.phonemize(text))
            except Exception:
                self._tokenizer = None
        return max(1, round(len(text) / _FALLBACK_CHARS_PER_PHONEME))


def _hard_split(text: str, max_phonemes: int, count) -> list[str]:
    """Last-resort word-boundary split of an oversized stretch of text."""
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and count(candidate) > max_phonemes:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_oversized(sentence: str, max_phonemes: int, count) -> list[str]:
    """Split an oversized sentence at clause boundaries, then word boundaries."""
    clauses = _CLAUSE_BREAK.split(sentence)
    chunks: list[str] = []
    current = ""
    for clause in clauses:
        candidate = f"{current} {clause}".strip()
        if current and count(candidate) > max_phonemes:
            chunks.append(current)
            current = ""
            if count(clause) > max_phonemes:
                chunks.extend(_hard_split(clause, max_phonemes, count))
            else:
                current = clause
        else:
            current = candidate
    if current:
        if count(current) > max_phonemes:
            chunks.extend(_hard_split(current, max_phonemes, count))
        else:
            chunks.append(current)
    return [c for c in chunks if c.strip()]


def chunk_text(
    text: str,
    max_phonemes: int = MAX_PHONEMES,
    count: PhonemeCounter | None = None,
) -> list[str]:
    """Split page text at sentence boundaries under a phoneme cap.

    Sentences are never split mid-sentence unless a single sentence alone
    overflows the cap; then clause boundaries, and finally word boundaries,
    are used. Quoted spans never merge with surrounding narration: a chunk
    boundary is inserted at every quote boundary so dialogue stays
    separable for diarization.
    """
    count = count or PhonemeCounter()
    chunks: list[str] = []
    current = ""
    for unit in _quote_units_preserving(text):
        for sentence in split_sentences(unit):
            merged = f"{current} {sentence}".strip()
            if count(merged) <= max_phonemes:
                current = merged
                continue
            if current:
                chunks.append(current)
                current = ""
            if count(sentence) <= max_phonemes:
                current = sentence
            else:
                chunks.extend(_split_oversized(sentence, max_phonemes, count))
        if current:
            # Never let a quoted unit merge with the next unit's text.
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _quote_units_preserving(text: str) -> list[str]:
    """Split at quote boundaries, keeping every character in order.

    Concatenating the returned units reproduces ``text`` exactly, so
    chunks built from separate units can still be space-joined to the
    original words. Quoted spans become their own units.
    """
    units: list[str] = []
    cursor = 0
    for match in _QUOTED_SPAN_RE.finditer(text):
        before = text[cursor : match.start()]
        if before.strip():
            units.append(before)
        units.append(match.group(0))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        units.append(tail)
    return units


def chunk_pages(
    pages_text: list[str],
    max_phonemes: int = MAX_PHONEMES,
    count: PhonemeCounter | None = None,
) -> list[list[str]]:
    count = count or PhonemeCounter()
    return [chunk_text(text, max_phonemes, count) for text in pages_text]
