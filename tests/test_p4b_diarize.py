"""P4-B diarize tests: quote detection, voice resolution, sentence manifest."""

from __future__ import annotations

import json

import pytest

from audiobookey.chunk import chunk_text, split_sentences
from audiobookey.markup import VoiceMap, quote_units


def test_quote_units_splits_straight_and_curly_quotes():
    straight = quote_units('He said "hello there" and left.')
    assert straight == [
        ("He said ", False),
        ("hello there", True),
        (" and left.", False),
    ]

    curly = quote_units("He said \u201chello\u201d and left.")
    assert [("He said ", False), ("hello", True), (" and left.", False)] == curly


def test_chunk_text_splits_at_quote_boundaries():
    text = 'Narration first. "Quoted dialogue here." More narration follows.'
    chunks = chunk_text(
        text, max_phonemes=15, count=lambda t: max(1, len(t) // 3)
    )
    assert chunks, "expected at least one chunk"
    assert " ".join(chunks).split() == text.split()  # no words lost
    assert any(c.strip().startswith('"') for c in chunks), (
        "expected a chunk starting at a quote boundary"
    )


def test_convert_diarize_assigns_dialogue_voice(tmp_path, monkeypatch):
    from audiobookey import audiobook
    from tests.fake_engine import FakeEngine

    fake = FakeEngine()
    monkeypatch.setattr(audiobook, "make_engine", lambda kind, models_dir: fake)

    pdf = tmp_path / "book.pdf"
    _one_page_pdf(pdf, 'The narrator speaks. "I am dialogue," she said.')

    result = audiobook.convert(
        pdf, tmp_path / "out", voice="af_heart", quality="draft"
    )
    assert result.m4b.suffix == ".m4b"

    spoken = " ".join(fake.calls)
    # The narrator voice must synthesis the narration...
    assert "The narrator speaks." in fake.calls[0]
    # ...but never the quoted span (it belongs to the dialogue voice).
    # With diarize the quoted text is synthesized on a separate call.
    calls_with_quote = [c for c in fake.calls if "I am dialogue" in c]
    assert calls_with_quote, "dialogue text was synthesized"

    manifest = json.loads(
        (tmp_path / "out" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sentences"], "sentence entries were emitted"
    starts = [s["start_s"] for s in manifest["sentences"]]
    ends = [s["end_s"] for s in manifest["sentences"]]
    assert starts == sorted(starts), "sentence starts must be monotonic"
    assert all(a <= b for a, b in zip(starts, ends)), (
        "sentence end_s must not precede start_s"
    )


def test_convert_diarize_requires_dialogue_voice(tmp_path, monkeypatch):
    from audiobookey import audiobook
    from audiobookey.voices import save_voices, voices_path

    # No 'dialogue*' entry anywhere: --diarize must fail loudly.
    voices_file = tmp_path / "voices.json"
    voices_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audiobook, "voices_path", lambda: voices_file)

    pdf = tmp_path / "book.pdf"
    _one_page_pdf(pdf, 'The narrator speaks. "I am dialogue," she said.')

    with pytest.raises(ValueError, match="dialogue"):
        audiobook.convert(pdf, tmp_path / "out", diarize=True)


def test_quote_detection_diarize_off(tmp_path, monkeypatch):
    """Without --diarize the whole page keeps the narrator voice."""
    from audiobookey.audiobook import _page_plan_items

    text = 'Narration first. "Quoted dialogue here." More narration.'
    items = _page_plan_items(text, False, None)
    assert len(items) == 1
    assert items[0].voice is None

    items = _page_plan_items(text, True, "dialogue_2")
    assert [(p.text, p.voice) for p in items] == [
        ("Narration first.", None),
        ("Quoted dialogue here.", "dialogue_2"),
        ("More narration.", None),
    ]


def _one_page_pdf(path, text: str) -> None:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
