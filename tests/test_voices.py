"""Voice manager unit tests: schema validation, fingerprints, resolution."""

from __future__ import annotations

import json

import numpy as np
import pytest

from audiobookey.audio import write_wav
from audiobookey.voices import (
    VoiceError,
    load_voices,
    resolve_voice,
    save_voices,
    validate_voice,
    voice_fingerprint,
)


def test_draft_voice_ok_and_final_needs_no_voice_id():
    validate_voice("narrator", {"engine": "draft", "voice_id": "af_heart"})
    validate_voice("clone1", {"engine": "final"})
    with pytest.raises(VoiceError):
        validate_voice("bad", {"engine": "draft"})  # missing voice_id
    with pytest.raises(VoiceError):
        validate_voice("bad2", {"engine": "final", "voice_id": "af_heart"})
    with pytest.raises(VoiceError):
        validate_voice("bad3", {"engine": "turbo"})
    with pytest.raises(VoiceError):
        validate_voice("bad name!", {"engine": "draft", "voice_id": "af_heart"})


def test_clone_sample_must_exist(tmp_path):
    missing = tmp_path / "nope.wav"
    with pytest.raises(VoiceError):
        validate_voice("v", {"engine": "final", "clone": str(missing)})


def test_seed_must_be_int():
    with pytest.raises(VoiceError):
        validate_voice("v", {"engine": "final", "seed": "abc"})


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "voices.json"
    voices = {"narrator": {"engine": "draft", "voice_id": "af_heart"}}
    save_voices(voices, path)
    assert load_voices(path) == voices


def test_load_invalid_file_raises_with_context(tmp_path):
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({"v": {"engine": "warp"}}), encoding="utf-8")
    with pytest.raises(VoiceError):
        load_voices(path)


def test_fingerprint_changes_with_config_and_clone_content(tmp_path):
    base = {"engine": "final"}
    a = voice_fingerprint(base)

    warmer = dict(base, temperature=0.9)
    b = voice_fingerprint(warmer)
    assert b != a

    sample = tmp_path / "sample.wav"
    write_wav(sample, np.zeros(240, dtype=np.float32), 24000)
    cfg1 = dict(base, clone=str(sample))
    f1 = voice_fingerprint(cfg1)

    sample2 = tmp_path / "sample.wav"
    write_wav(sample2, np.ones(240, dtype=np.float32), 24000)  # same path, new content
    f2 = voice_fingerprint(cfg1)
    assert f2 != f1, "clone content change must change the fingerprint"


def test_resolve_voice_falls_back_to_raw_kokoro_id_for_draft(tmp_path):
    path = tmp_path / "voices.json"
    save_voices({"narrator": {"engine": "draft", "voice_id": "af_heart"}}, path)
    assert resolve_voice("narrator", "draft", path)["voice_id"] == "af_heart"
    # unknown name + draft quality: treated as a raw kokoro voice id
    assert resolve_voice("am_michael", "draft", path) == {
        "engine": "draft", "voice_id": "am_michael"
    }
    # unknown name + final quality: error
    with pytest.raises(VoiceError):
        resolve_voice("am_michael", "final", path)
