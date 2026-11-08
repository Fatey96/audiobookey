"""Per-page text cleanup.

Order of operations (per the P1 brief):
1. Strip running headers/footers detected statistically across pages, plus bare
   page numbers, in the edge zones (first/last few lines) of each page.
2. Rejoin hyphenated linebreaks and de-wrap hard linebreaks into paragraphs
   (blank line = paragraph break).
3. Pages with no extractable text are marked skipped.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Lines considered part of the header/footer zone at the top and bottom of a page.
_EDGE_LINES = 3

_BARE_PAGE_NUMBER = re.compile(
    r"^\s*[-\u2013\u2014~]*(\d{1,4}|[ivxlcdm]{1,7})[-\u2013\u2014~]*\s*$",
    re.IGNORECASE,
)

_NON_ALNUM = re.compile(r"[^a-z#]+")


@dataclass
class CleanedPage:
    number: int  # 1-based PDF page number
    text: str
    skipped: bool


def normalize_line(line: str) -> str:
    """Normalization used to match RunningHeader/Footer lines across pages.

    Lowercases, collapses whitespace, replaces digits with '#' (so 'Page 3'
    and 'Page 4' match) and strips punctuation.
    """
    key = line.strip().lower()
    key = key.translate(str.maketrans("0123456789", "##########"))
    key = _NON_ALNUM.sub(" ", key)
    return key.strip()


def _edge_indexes(lines: list[str]) -> set[int]:
    """Indexes of the first and last non-empty lines of a page."""
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return set()
    head = set(nonempty[:_EDGE_LINES])
    tail = set(nonempty[-_EDGE_LINES:])
    return head | tail


def find_running_lines(raw_pages: list[str], threshold: float = 0.5) -> set[str]:
    """Return normalized line keys that recur in page edge zones on enough pages.

    A line counts as a running header/footer if its normalized form appears in
    the edge zone of at least `threshold` of the pages that have any text (and
    at least 2 pages, so tiny documents are not stripped blindly).
    """
    pages_lines = [raw.splitlines() for raw in raw_pages]
    nonempty = [lines for lines in pages_lines if any(line.strip() for line in lines)]
    if len(nonempty) < 2:
        return set()

    counts: dict[str, int] = {}
    for lines in nonempty:
        edges = _edge_indexes(lines)
        seen: set[str] = set()
        for idx in edges:
            key = normalize_line(lines[idx])
            if key and key not in seen:
                seen.add(key)
                counts[key] = counts.get(key, 0) + 1

    needed = max(2, math.ceil(threshold * len(nonempty)))
    return {key for key, count in counts.items() if count >= needed}


def _strip_edges(lines: list[str], running: set[str]) -> list[str]:
    """Drop running header/footer lines and bare page numbers in edge zones."""
    edges = _edge_indexes(lines)
    kept: list[str] = []
    for idx, line in enumerate(lines):
        if idx in edges:
            stripped = line.strip()
            if _BARE_PAGE_NUMBER.match(stripped):
                continue
            if normalize_line(stripped) in running:
                continue
        kept.append(line)
    return kept


def _join_paragraph(lines: list[str]) -> str:
    """De-wrap one paragraph: rejoin hyphenated linebreaks, join the rest."""
    text = ""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not text:
            text = s
        elif text.endswith("-") and s[:1].isalpha():
            # 'exam-' + 'ple' -> 'example'
            text = text[:-1] + s
        else:
            text += " " + s
    return text


def flow_text(lines: list[str]) -> str:
    """Merge hard-wrapped lines into paragraphs; blank lines are breaks."""
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return "\n\n".join(_join_paragraph(p) for p in paragraphs if p)


def clean_page(raw: str, running: set[str]) -> str:
    lines = raw.splitlines()
    stripped = _strip_edges(lines, running)
    if not any(line.strip() for line in stripped):
        # Safety: header stripping must never empty out a page (this can only
        # happen when every line of a page looks like a running header).
        stripped = lines
    return flow_text(stripped)


def clean_pages(raw_pages: list[str], threshold: float = 0.5) -> list[CleanedPage]:
    """Clean every page; pages with no text left are marked skipped."""
    running = find_running_lines(raw_pages, threshold=threshold)
    cleaned: list[CleanedPage] = []
    for i, raw in enumerate(raw_pages, start=1):
        text = clean_page(raw, running) if raw.strip() else ""
        cleaned.append(
            CleanedPage(number=i, text=text, skipped=not text.strip())
        )
    return cleaned
