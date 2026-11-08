"""Markup unit tests: pause markers, [voice:] regions, paragraph pauses."""

from __future__ import annotations

import pytest

from audiobookey import markup
from audiobookey.markup import (
    MarkupError,
    PauseSegment,
    SpeechSegment,
    VoiceMap,
    split_segments,
)


def test_pause_marker_splits_text():
    segs = split_segments("Before the pause.[pause:1.5]After the pause.")
    assert segs == [
        SpeechSegment("Before the pause.", None),
        PauseSegment(1.5),
        SpeechSegment("After the pause.", None),
    ]


def test_pause_marker_never_lands_inside_a_chunk_source():
    # The segment texts carry no markup at all.
    segs = split_segments("One. [pause:2] Two.")
    speech = [s.text for s in segs if isinstance(s, SpeechSegment)]
    assert all("pause" not in t for t in speech)
    assert "One." in speech and "Two." in speech


def test_voice_region_tags_following_text():
    segs = split_segments("Narration.[voice:bob]Bob speaks now.[/voice]Narration again.")
    assert segs == [
        SpeechSegment("Narration.", None),
        SpeechSegment("Bob speaks now.", "bob"),
        SpeechSegment("Narration again.", None),
    ]


def test_unclosed_voice_region_runs_to_end():
    segs = split_segments("Start.[voice:carol]Carol talks a lot")
    assert segs == [
        SpeechSegment("Start.", None),
        SpeechSegment("Carol talks a lot", "carol"),
    ]


def test_nested_voice_region_rejected():
    with pytest.raises(MarkupError):
        split_segments("[voice:a]x[voice:b]y")


def test_empty_segments_are_dropped():
    segs = split_segments("[pause:1][pause:2]")
    assert segs == [PauseSegment(1.0), PauseSegment(2.0)]


def test_voice_map_first_matching_pattern_wins():
    vm = VoiceMap({"^\\s*ANA:": "ana", "question\\?": "bob"})
    assert vm.match("ANA: hello") == "ana"
    assert vm.match("why not? was that a question? ANA: nope") == "bob"
    assert vm.match("nothing here") is None


def test_voice_map_invalid_regex_rejected(tmp_path):
    import json

    p = tmp_path / "map.json"
    p.write_text(json.dumps({"([bad": "x"}), encoding="utf-8")
    with pytest.raises(MarkupError):
        VoiceMap.from_file(p)


def test_paragraph_pause_constant():
    assert markup.PARAGRAPH_PAUSE == 0.6
