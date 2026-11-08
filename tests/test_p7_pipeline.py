"""P7-B: a page becomes playable the moment it is synthesized.

convert() must assemble and announce each page inside the synthesis loop:
after a page's chunk loop finishes, the page WAV is built immediately and
announced via ``reporter.page_ready(page, index, total_pages, wav,
duration_s, sentences)``. These tests use the fake engine (no GPU, no
models) and a small recording reporter stub — ``reporter.py`` itself is
owned by another agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

from audiobookey.audio import ffprobe_info, read_wav_duration
from audiobookey.reporter import QuietReporter

from .fake_engine import FakeEngine


class RecordingReporter(QuietReporter):
    """Records page_ready calls; everything else stays silent."""

    def __init__(self) -> None:
        self.page_ready_events: list[dict] = []

    def page_ready(
        self,
        page: int,
        index: int,
        total_pages: int,
        wav,
        duration_s: float,
        sentences: list[dict],
    ) -> None:
        self.page_ready_events.append(
            {
                "page": page,
                "index": index,
                "total_pages": total_pages,
                "wav": Path(wav),
                "duration_s": duration_s,
                "sentences": [dict(s) for s in sentences],
            }
        )


def make_pdf(path: Path, pages: int = 3) -> Path:
    """Multi-sentence pages; insert_textbox keeps text past the first line."""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        text = (
            f"Page {i + 1} opens the tale here. "
            + " ".join(
                f"Entry {n} kept its steady song through the trees."
                for n in range(1, 7)
            )
            + " Night settled in at last."
        )
        page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


def _convert(out_dir: Path, pdf: Path) -> tuple:
    """Run convert() on out_dir (reusing its chunk cache) with the fake engine."""
    from audiobookey import audiobook as ab

    fake = FakeEngine()
    reporter = RecordingReporter()
    result = ab.convert(pdf, out_dir, engine=fake, reporter=reporter)
    return result, reporter, fake


@pytest.fixture()
def three_page_pdf(tmp_path):
    return make_pdf(tmp_path / "book.pdf", pages=3)


def test_page_ready_per_page_in_order(three_page_pdf, tmp_path):
    result, reporter, _ = _convert(tmp_path / "out", three_page_pdf)

    events = reporter.page_ready_events
    assert len(events) == result.chapters == 3
    # exactly one event per usable page, in ascending index order
    assert [e["index"] for e in events] == [1, 2, 3]
    # page numbers match the PDF's page numbering
    assert [e["page"] for e in events] == [1, 2, 3]
    assert all(e["total_pages"] == 3 for e in events)
    for event in events:
        assert event["duration_s"] > 0
        assert event["wav"].is_file(), event["wav"]
        # the reported WAV's real duration matches the reported one
        assert (
            abs(read_wav_duration(event["wav"]) - event["duration_s"]) <= 0.05
        )


def test_page_ready_durations_sum_to_m4b_duration(three_page_pdf, tmp_path):
    result, reporter, _ = _convert(tmp_path / "out", three_page_pdf)

    total = sum(e["duration_s"] for e in reporter.page_ready_events)
    # chapters[-1][1] == ConvertResult.duration_seconds
    assert abs(total - result.duration_seconds) <= 1e-3

    info = ffprobe_info(result.m4b)
    assert len(info.get("chapters", [])) == 3
    titles = [c["tags"]["title"] for c in info["chapters"]]
    assert titles == ["Page 1", "Page 2", "Page 3"]


def test_sentences_are_page_local(three_page_pdf, tmp_path):
    _result, reporter, _ = _convert(tmp_path / "out", three_page_pdf)

    for event in reporter.page_ready_events:
        sentences = event["sentences"]
        assert sentences, "every page must publish its sentences"
        starts = [s["start_s"] for s in sentences]
        assert all(s >= 0.0 for s in starts)
        assert all(
            s["end_s"] <= event["duration_s"] + 1e-3 for s in sentences
        )
        assert starts == sorted(starts), "sentence starts must ascend"
        assert all(s["text"].strip() for s in sentences), (
            "no empty sentence texts"
        )
        # page-local offsets: the page starts at zero (catches a global
        # cursor leaking into the per-page offsets)
        assert starts[0] < 0.01


def test_second_run_with_warm_cache_publishes_every_page_again(
    three_page_pdf, tmp_path
):
    out_dir = tmp_path / "out"
    first_result, first_reporter, first_fake = _convert(out_dir, three_page_pdf)
    assert first_fake.calls, "sanity: first run synthesized chunks"

    second_result, second_reporter, second_fake = _convert(out_dir, three_page_pdf)
    assert second_fake.calls == [], "cached re-run synthesizes nothing"

    second_events = second_reporter.page_ready_events
    assert len(second_events) == len(first_reporter.page_ready_events) == 3
    assert [e["page"] for e in second_events] == [1, 2, 3]
    assert [e["index"] for e in second_events] == [1, 2, 3]
    assert second_result.cache_misses == 0


def test_reader_manifest_unchanged(three_page_pdf, tmp_path):
    _result, reporter, _ = _convert(tmp_path / "out", three_page_pdf)

    manifest = json.loads(
        (tmp_path / "out" / "manifest.json").read_text(encoding="utf-8")
    )
    entries = manifest["sentences"]
    expected = [
        (e["page"], s["start_s"], s["end_s"], s["text"])
        for e in reporter.page_ready_events
        for s in e["sentences"]
    ]
    # one entry per sentence, in the same order, now with global offsets
    assert len(entries) == len(expected)
    for entry, (page, local_start, local_end, text) in zip(entries, expected):
        assert entry["page"] == page
        assert entry["text"] == text
        assert entry["end_s"] >= entry["start_s"] >= 0.0
    # offsets are global: they grow across pages monotonically
    starts = [entry["start_s"] for entry in entries]
    assert starts == sorted(starts)
    # the first sentence of page 2 starts after page 1's audio ends
    page2_first = next(e for e in entries if e["page"] == 2)
    page1_last = [e for e in entries if e["page"] == 1][-1]
    assert page2_first["start_s"] >= page1_last["end_s"] - 1e-3
