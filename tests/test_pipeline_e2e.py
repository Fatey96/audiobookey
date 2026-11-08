"""End-to-end pipeline test: synthetic PDF -> wavs -> m4b with 3 chapters.

Uses FakeEngine so no model download or GPU is required; ffprobe verifies the
m4b really has 3 chapters per page.
"""

from __future__ import annotations

import pymupdf
import pytest

from audiobookey.audio import ffprobe_info

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


def test_convert_synthetic_pdf_makes_m4b_with_three_chapters(
    tmp_path, three_page_pdf, monkeypatch, capsys
):
    out_dir = tmp_path / "out"
    fake = FakeEngine()

    monkeypatch.setenv("AUDIOBOOKEY_MODELS_DIR", str(tmp_path / "nonexistent"))
    # Engine is injected; models dir never touched.
    from audiobookey import audiobook as ab

    result = ab.convert(
        three_page_pdf,
        out_dir,
        voice="af_heart",
        speed=1.0,
        engine=fake,
    )

    assert result.chapters == 3
    assert result.m4b.is_file()
    assert result.mp3.is_file()
    assert result.skipped_pages == []
    assert result.cache_hits == 0 and result.cache_misses == result.total_chunks
    assert len(fake.calls) == result.total_chunks

    info = ffprobe_info(result.m4b)
    assert len(info.get("chapters", [])) == 3
    titles = [c["tags"]["title"] for c in info["chapters"]]
    assert titles == ["Page 1", "Page 2", "Page 3"]
    # chapter starts are monotonically laid out over the book
    starts = [float(c["start_time"]) for c in info["chapters"]]
    assert starts == sorted(starts)
    assert float(info["format"]["duration"]) > 0


def test_second_run_hits_100_percent_cache(tmp_path, three_page_pdf):
    from audiobookey import audiobook as ab

    out_dir = tmp_path / "out"
    fake = FakeEngine()
    ab.convert(three_page_pdf, out_dir, engine=fake)
    assert fake.calls  # sanity: some synthesis happened on the first run

    fake2 = FakeEngine()
    result2 = ab.convert(three_page_pdf, out_dir, engine=fake2)
    assert fake2.calls == [], "nothing should be synthesized on a cached re-run"
    assert result2.cache_misses == 0
    assert result2.cache_hit_rate == 1.0
