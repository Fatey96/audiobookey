"""P9 --cache-root: one durable chunk cache shared across --out dirs."""

from __future__ import annotations

import pymupdf
import pytest

from audiobookey.cache import book_id

from .fake_engine import FakeEngine


@pytest.fixture()
def three_page_pdf(tmp_path):
    path = tmp_path / "book.pdf"
    doc = pymupdf.open()
    for i in range(1, 4):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"This is page number {i}.\nIt has a couple of sentences so that "
            "the pipeline has real work to do. The end of the page follows.",
            fontsize=12,
        )
    doc.save(str(path))
    doc.close()
    return path


def test_cache_root_chunks_are_hits_for_a_different_out_dir(
    tmp_path, three_page_pdf, monkeypatch
):
    """Chunks written under --cache-root are hits for a second run whose
    --out dir differs: the cache follows the root, not the output dir, so
    resume survives fresh per-job output directories.
    """
    from audiobookey import audiobook as ab

    monkeypatch.setenv("AUDIOBOOKEY_MODELS_DIR", str(tmp_path / "nonexistent"))

    cache_root = tmp_path / "durable-cache"
    out_a = tmp_path / "job-a"
    out_b = tmp_path / "job-b"
    bid = book_id(three_page_pdf)

    fake = FakeEngine()
    first = ab.convert(
        three_page_pdf, out_a, engine=fake, cache_root=cache_root
    )
    assert first.cache_hits == 0
    assert first.cache_misses == first.total_chunks

    chunk_dir = cache_root / bid / "chunks"
    assert chunk_dir.is_dir(), "chunk cache must live under --cache-root"
    assert list(chunk_dir.glob("*.wav")), "chunks must be under --cache-root"
    assert (cache_root / bid / "manifest.json").is_file()
    assert not (out_a / "cache").exists(), "--out must not hold the cache"
    assert (out_a / "work" / bid).is_dir(), "work dir stays under out_dir"

    fake2 = FakeEngine()
    second = ab.convert(
        three_page_pdf, out_b, engine=fake2, cache_root=cache_root
    )
    assert fake2.calls == [], "nothing should be synthesized on a cached re-run"
    assert second.cache_misses == 0
    assert second.cache_hits == second.total_chunks
    assert second.total_chunks == first.total_chunks
    assert second.cache_hit_rate == 1.0
    assert second.m4b.is_file() and second.mp3.is_file()
    assert not (out_b / "cache").exists(), "--out must not hold the cache"
    assert (out_b / "work" / bid).is_dir(), "work dir stays under out_dir"


def test_default_cache_stays_under_out_dir(
    tmp_path, three_page_pdf, monkeypatch
):
    """No --cache-root: chunks land under <out>/cache/<book-id>/chunks and a
    plain re-run into the same out dir hits 100% cache, exactly as before.
    """
    from audiobookey import audiobook as ab

    monkeypatch.setenv("AUDIOBOOKEY_MODELS_DIR", str(tmp_path / "nonexistent"))

    out_dir = tmp_path / "out"
    bid = book_id(three_page_pdf)

    fake = FakeEngine()
    first = ab.convert(three_page_pdf, out_dir, engine=fake)
    chunk_dir = out_dir / "cache" / bid / "chunks"
    assert chunk_dir.is_dir(), "no flag: chunk cache stays under <out>/cache"
    assert list(chunk_dir.glob("*.wav"))
    assert first.cache_hits == 0
    assert first.cache_misses == first.total_chunks

    fake2 = FakeEngine()
    second = ab.convert(three_page_pdf, out_dir, engine=fake2)
    assert fake2.calls == []
    assert second.cache_misses == 0
    assert second.cache_hit_rate == 1.0
