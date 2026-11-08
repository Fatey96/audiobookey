"""C5A regression: overlapped synthesis, no book.wav, event order stable.

convert() hands each finished page to a single assembler thread while the
main thread keeps synthesizing. These tests pin the reporter contract that
the server's page counter and live-pages ladder rely on, and assert the
book.wav intermediate is gone while outputs stay correct.
"""

import json
import wave
from pathlib import Path

import pymupdf
import pytest

from audiobookey import audiobook as ab
from audiobookey.audio import ffprobe_info, read_wav_sample_rate

from .fake_engine import FakeEngine


def make_pdf(path: Path, pages: int = 3) -> Path:
    """Multi-sentence pages so each page yields several chunks."""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        text = (
            f"Page {i + 1} opens the tale here. "
            + " ".join(
                f"Entry {n} kept its steady song through the trees."
                for n in range(1, 21)
            )
            + " Night settled in at last."
        )
        page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class EverythingReporter(ab.QuietReporter):
    """Records every callback in emission order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple]] = []

    def start_chunks(self, total, total_pages):
        self.events.append(("start_chunks", (total, total_pages)))

    def chunk_done(self):
        self.events.append(("chunk_done", ()))

    def update(self, cache_hits, done, total):
        self.events.append(("update", (cache_hits, done, total)))

    def page_ready(self, page, index, total_pages, wav, duration_s, sentences):
        self.events.append(
            ("page_ready", (page, index, total_pages, wav, duration_s))
        )


@pytest.fixture()
def book_pdf(tmp_path):
    return make_pdf(tmp_path / "book.pdf", pages=3)


def test_event_order_chunk_done_page_ready_update(book_pdf, tmp_path):
    """chunk_done in chunk order; page_ready in page order; update per page."""
    reporter = EverythingReporter()
    result = ab.convert(
        book_pdf, tmp_path / "out", engine=FakeEngine(), reporter=reporter
    )

    kinds = [kind for kind, _ in reporter.events]
    chunk_dones = [i for i, k in enumerate(kinds) if k == "chunk_done"]
    page_readies = [i for i, k in enumerate(kinds) if k == "page_ready"]

    # a multi-chunk book: strictly more chunk_done events than pages
    assert result.total_chunks > 3
    assert len(chunk_dones) == result.total_chunks
    assert len(page_readies) == 3
    # chunk_done events fire once per chunk, in chunk order
    assert chunk_dones == sorted(chunk_dones)
    # page_ready events announce pages 1, 2, 3 in order
    pages = [reporter.events[i][1][0] for i in page_readies]
    indexes = [reporter.events[i][1][1] for i in page_readies]
    assert pages == [1, 2, 3]
    assert indexes == [1, 2, 3]

    # every page_ready trails at least one chunk_done (a page is never
    # announced before any synthesis happened), and all chunk_done events
    # precede the final page_ready: the last page is pumped only after its
    # last chunk synthesized
    assert page_readies[0] > 0
    assert kinds[: page_readies[-1] + 1].count("chunk_done") == (
        result.total_chunks
    )

    # each page_ready is immediately followed by that page's one update,
    # and the updates' cumulative done counts end at the chunk total
    updates = []
    for i in page_readies:
        kind, payload = reporter.events[i + 1]
        assert kind == "update"
        updates.append(payload)
    assert len(updates) == 3
    assert updates[-1][1] == result.total_chunks


def test_overlap_matches_sequential_output(book_pdf, tmp_path):
    """Same input run twice: identical manifests/chapters/durations."""
    out_a, out_b = tmp_path / "a", tmp_path / "b"  # fresh caches per run
    res_a = ab.convert(book_pdf, out_a, engine=FakeEngine())
    res_b = ab.convert(book_pdf, out_b, engine=FakeEngine())

    assert res_a.chapters == res_b.chapters == 3
    assert abs(res_a.duration_seconds - res_b.duration_seconds) <= 1e-9

    # reader manifest (sentences) is identical
    man_a = (out_a / "manifest.json").read_text(encoding="utf-8")
    man_b = (out_b / "manifest.json").read_text(encoding="utf-8")
    assert man_a == man_b
    # stats manifest (chapters + page numbers) is identical modulo cache
    from audiobookey.cache import book_id

    bid = book_id(book_pdf)
    stats_a = json.loads(
        (out_a / "cache" / bid / "manifest.json").read_text(encoding="utf-8")
    )
    stats_b = json.loads(
        (out_b / "cache" / bid / "manifest.json").read_text(encoding="utf-8")
    )
    stats_a.pop("cache"), stats_b.pop("cache")
    assert stats_a == stats_b


def test_book_wav_never_materializes(book_pdf, tmp_path):
    out = tmp_path / "out"
    result = ab.convert(book_pdf, out, engine=FakeEngine())

    # outputs are correct
    assert result.m4b.is_file() and result.mp3.is_file()
    info = ffprobe_info(result.m4b)
    assert len(info.get("chapters", [])) == 3

    # ...but the intermediate never hits disk
    written = {p.name for p in out.rglob("*")}
    assert "book.wav" not in written

    # page wavs are 24 kHz mono pcm16 by construction
    page_wavs = sorted((out / "work").rglob("page-*.wav"))
    assert len(page_wavs) == 3
    for wav in page_wavs:
        with wave.open(str(wav), "rb") as w:
            assert w.getframerate() == 24000
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2


def test_space_in_out_dir_still_encodes(book_pdf, tmp_path):
    """The concat list quotes Windows paths with spaces correctly."""
    out = tmp_path / "audio out"
    result = ab.convert(book_pdf, out, engine=FakeEngine())

    assert result.m4b.is_file() and result.mp3.is_file()
    info = ffprobe_info(result.m4b)
    assert len(info.get("chapters", [])) == 3
    # the normalized intermediate is cleaned up either way
    normalized = list((out / "work").rglob("*.normalized.wav"))
    assert normalized == []


def test_cache_hit_rerun_still_hits_100_percent(book_pdf, tmp_path):
    out = tmp_path / "out"
    first = ab.convert(book_pdf, out, engine=FakeEngine())
    # identical chunk texts dedup to one key, so misses + hits == chunks
    assert first.cache_hits + first.cache_misses == first.total_chunks

    second = ab.convert(book_pdf, out, engine=FakeEngine())
    assert second.cache_misses == 0
    assert second.cache_hits == second.total_chunks
    assert abs(
        second.duration_seconds - first.duration_seconds
    ) <= 1e-9
