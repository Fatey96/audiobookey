"""P5B tests: the /api/voices ``label`` field and POST /api/voices/{name}/preview.

No TTS runs in these tests: the preview synthesis call is monkeypatched, so
neither the unknown-voice nor the cache-hit path needs a GPU or the models/
directory.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from audiobookey.server import previews


@pytest.fixture()
def server_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(data_dir))
    monkeypatch.chdir(tmp_path)  # keep find_data_dir()/find_models_dir() away
    data_dir.mkdir()
    return {"data_dir": data_dir}


def make_client():
    from audiobookey.server.api import create_app

    return TestClient(create_app())


class StubSynth:
    """Stands in for previews.synthesize_preview; counts every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, quality, voice_config, models_dir=None):
        self.calls.append((quality, dict(voice_config)))
        n = 2400  # 0.1 s of 24 kHz tone
        t = np.arange(n, dtype=np.float32) / 24000.0
        return 0.2 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32), 24000


# -- GET /api/voices: label field ------------------------------------------------


def test_voices_label_present_on_all_items(server_env, tmp_path):
    from audiobookey.voices import save_voices

    clone_wav = tmp_path / "me.wav"
    clone_wav.write_bytes(b"RIFF")  # path existence is all validation needs
    save_voices(
        {"narrator": {"engine": "draft", "voice_id": "am_michael"}},
        path=server_env["data_dir"] / "voices.json",
    )
    with make_client() as client:
        voices = client.get("/api/voices").json()
        assert voices, "expected at least the draft defaults"
        for item in voices:
            assert item["label"] == item["name"].replace("_", " ")
        by_name = {v["name"]: v for v in voices}
        assert by_name["af_heart"]["label"] == "af heart"
        assert by_name["am_fenrir"]["label"] == "am fenrir"
        assert by_name["narrator"]["label"] == "narrator"


# -- POST /api/voices/{name}/preview -----------------------------------------------


def test_preview_unknown_voice_returns_404(server_env):
    with make_client() as client:
        resp = client.post("/api/voices/definitely_not_a_voice/preview")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "unknown voice"}


def test_preview_happy_path_and_cache_hit(server_env, monkeypatch):
    stub = StubSynth()
    monkeypatch.setattr(previews, "synthesize_preview", stub)
    with make_client() as client:
        first = client.post("/api/voices/af_heart/preview")
        assert first.status_code == 200, first.text
        assert first.headers["content-type"] == "audio/wav"
        assert first.content[:4] == b"RIFF"
        assert len(stub.calls) == 1
        quality, cfg = stub.calls[0]
        assert quality == "draft"
        assert cfg == {"engine": "draft", "voice_id": "af_heart"}

        cache_dir = server_env["data_dir"] / "outputs" / "previews"
        cached = list(cache_dir.glob("*.wav"))
        assert len(cached) == 1
        assert cached[0].read_bytes() == first.content

        second = client.post("/api/voices/af_heart/preview")
        assert second.status_code == 200
        assert second.headers["content-type"] == "audio/wav"
        assert second.content == first.content
        # cache hit: the synthesis stub was NOT called again
        assert len(stub.calls) == 1


def test_preview_configured_voice_uses_its_engine(server_env, monkeypatch):
    from audiobookey.voices import save_voices

    clone_wav = server_env["data_dir"] / "me.wav"
    clone_wav.write_bytes(b"RIFF")
    save_voices(
        {"cloned": {"engine": "final", "clone": str(clone_wav)}},
        path=server_env["data_dir"] / "voices.json",
    )
    stub = StubSynth()
    monkeypatch.setattr(previews, "synthesize_preview", stub)
    with make_client() as client:
        resp = client.post("/api/voices/cloned/preview")
        assert resp.status_code == 200, resp.text
        assert stub.calls[0][0] == "final"
        assert stub.calls[0][1]["engine"] == "final"


def test_preview_synthesis_failure_returns_500_json(server_env, monkeypatch):
    def boom(quality, voice_config, models_dir=None):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(previews, "synthesize_preview", boom)
    with make_client() as client:
        resp = client.post("/api/voices/af_heart/preview")
        assert resp.status_code == 500
        assert "model exploded" in resp.json()["detail"]
