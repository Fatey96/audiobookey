"""P6B quality-tier tests: low/medium/high accepted server-side (draft/final
kept as backwards-compatible aliases), engine resolution of the tier map, and
the best-vs-standard encode tier threaded through the final encode — without
real synthesis or GPU.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from audiobookey import audiobook as ab
from audiobookey import cli as cli_mod
from audiobookey import engine_pool
from audiobookey.audio import ffprobe_info, write_wav
from audiobookey.server import jobs as jobs_mod
from audiobookey.server.api import QUALITY_VALUES, create_app
from audiobookey.voices import resolve_voice, save_voices

from .fake_engine import FakeEngine

ACCEPTED_QUALITIES = ("low", "medium", "high", "draft", "final")
BOGUS_QUALITY_MESSAGE = "quality must be draft, final, low, medium or high"


@pytest.fixture()
def server_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(data_dir))
    monkeypatch.chdir(tmp_path)  # keep find_data_dir() away from the repo data/

    pdf = tmp_path / "book.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "This is page number 1.\nIt has a couple of sentences so that the "
        "pipeline has real work to do. The end of the page follows.",
        fontsize=12,
    )
    doc.save(str(pdf))
    doc.close()
    return {"data_dir": data_dir, "pdf": pdf}


@pytest.fixture()
def capture_jobs(monkeypatch):
    """Replace the convert worker with a recorder.

    Posted jobs are captured and completed without any engine ever being
    constructed or consulted: no synthesis, no model load, no GPU.
    """
    captured = []

    def fake_run_job(self, task):
        captured.append(
            {
                "job_id": task["job"]["id"],
                "quality": task["quality"],
                "voice": task["voice"],
            }
        )
        task["job"]["status"] = "done"

    monkeypatch.setattr(jobs_mod.JobManager, "_run_job", fake_run_job)
    return captured


def poll_captured(captured, job_id: str, timeout_s: float = 5.0) -> dict:
    """The recorder runs on the worker thread; poll until our job is recorded."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for entry in captured:
            if entry["job_id"] == job_id:
                return entry
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} was never recorded by the worker")


# -- /api/convert + /api/convert/batch: all tiers accepted, bogus 400 --------


@pytest.mark.parametrize("quality", ACCEPTED_QUALITIES)
def test_convert_accepts_all_quality_values(server_env, capture_jobs, quality):
    with TestClient(create_app()) as client:
        with open(server_env["pdf"], "rb") as f:
            resp = client.post(
                "/api/convert",
                files={"pdf": ("book.pdf", f, "application/pdf")},
                data={"voice": "af_heart", "quality": quality, "gap": "0.2"},
            )
        assert resp.status_code == 201, resp.text
        job = resp.json()

        entry = poll_captured(capture_jobs, job["id"])
        assert entry["quality"] == quality  # passed through unaliased
        assert entry["voice"] == "af_heart"


@pytest.mark.parametrize("quality", ACCEPTED_QUALITIES)
def test_convert_batch_accepts_all_quality_values(
    server_env, capture_jobs, quality
):
    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/convert/batch",
            json={"pdf_paths": [str(server_env["pdf"])]},
        )
        assert resp.status_code == 201, resp.text

        resp = client.post(
            "/api/convert/batch",
            json={"pdf_paths": [str(server_env["pdf"])], "quality": quality},
        )
        assert resp.status_code == 201, resp.text
        job = resp.json()

        entry = poll_captured(capture_jobs, job["id"])
        assert entry["quality"] == quality


def test_convert_rejects_bogus_quality(server_env, capture_jobs):
    with TestClient(create_app()) as client:
        with open(server_env["pdf"], "rb") as f:
            resp = client.post(
                "/api/convert",
                files={"pdf": ("book.pdf", f, "application/pdf")},
                data={"quality": "turbo"},
            )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"] == BOGUS_QUALITY_MESSAGE

        resp = client.post(
            "/api/convert/batch",
            json={"pdf_paths": [str(server_env["pdf"])], "quality": "ultra"},
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"] == BOGUS_QUALITY_MESSAGE

        assert capture_jobs == [], "no job may be recorded for a bogus quality"


# -- engine resolution: tiers alias onto engine tiers, cfg stays stable ------

def test_cli_quality_tiers_alias_onto_engine_tiers():
    assert cli_mod.QUALITY_ENGINE_ALIASES == {
        "low": "draft",
        "medium": "final",
        "high": "final",
    }


def test_rejected_engine_tier_via_make_engine(monkeypatch, tmp_path):
    created = []

    class Recorded:
        def __init__(self, tag, models_dir):
            self.tag = tag
            self.engine_version = f"recorded=={tag}"

    def spawn(tag):
        def ctor(models_dir):
            created.append(tag)
            return Recorded(tag, models_dir)

        return ctor

    # Constructors are patched, so no model is ever loaded: resolution only.
    monkeypatch.setattr(engine_pool, "KokoroTTSEngine", spawn("draft"))
    monkeypatch.setattr(engine_pool, "ChatterboxEngine", spawn("final"))

    models_dir = str(tmp_path / "models")
    for quality, tag in (
        ("low", "draft"),
        ("medium", "final"),
        ("high", "final"),
        ("draft", "draft"),
        ("final", "final"),
    ):
        engine_tier = cli_mod.QUALITY_ENGINE_ALIASES.get(quality, quality)
        engine = engine_pool.make_engine(engine_tier, models_dir)
        assert engine.tag == tag
    assert created == ["draft", "final", "final", "draft", "final"]
    with pytest.raises(ValueError):
        engine_pool.make_engine("turbo", models_dir)


def test_resolve_voice_engine_for_all_tiers(tmp_path):
    clone_wav = str(tmp_path / "clone.wav")
    write_wav(clone_wav, np.zeros(1200, dtype=np.float32), 24000)
    voices_path = tmp_path / "voices.json"
    save_voices({"clone1": {"engine": "final", "clone": clone_wav}}, voices_path)

    # The CLI resolves the engine alias before convert() calls resolve_voice,
    # so resolve_voice only ever sees the draft/final engine tiers.
    for quality in ("medium", "high", "final"):
        cfg = resolve_voice("clone1", quality, voices_path)
        assert cfg["engine"] == "final"
    cfg = resolve_voice("af_heart", "draft", voices_path)
    assert cfg == {"engine": "draft", "voice_id": "af_heart"}


# -- encode tier: best bumps the output bitrates over standard ---------------


@pytest.fixture()
def one_page_pdf(tmp_path):
    pdf = tmp_path / "book.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72), "A single page with a couple of sentences to convert."
    )
    doc.save(str(pdf))
    doc.close()
    return pdf


@pytest.fixture()
def final_voice(tmp_path, monkeypatch):
    """A configured final-engine voice so convert()'s voice resolution sees it.

    convert() with quality=final rejects bare Kokoro ids by design (P2), so the
    encode-tier tests need a voices.json entry; find_data_dir() prefers
    cwd/data, so chdir into tmp_path with a data/ voices.json planted there.
    """
    clone_wav = str(tmp_path / "clone.wav")
    write_wav(clone_wav, np.zeros(1200, dtype=np.float32), 24000)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    save_voices(
        {"clone1": {"engine": "final", "clone": clone_wav}},
        data_dir / "voices.json",
    )
    monkeypatch.chdir(tmp_path)
    return "clone1"


@pytest.mark.parametrize("ext", ["m4b", "mp3"])
def test_encode_tier_best_bitrate_above_standard(
    tmp_path, one_page_pdf, ext, final_voice
):
    best_out = tmp_path / "best"
    standard_out = tmp_path / "standard"

    ab.convert(
        one_page_pdf,
        best_out,
        voice=final_voice,
        quality="final",
        encode_tier="best",
        engine=FakeEngine(),
    )
    ab.convert(
        one_page_pdf,
        standard_out,
        voice=final_voice,
        quality="final",
        engine=FakeEngine(),
    )

    best = best_out / f"{one_page_pdf.stem}.{ext}"
    standard = standard_out / f"{one_page_pdf.stem}.{ext}"
    assert best.is_file() and standard.is_file(), "both tiers must encode"

    env = os.environ
    best_rate = int(ffprobe_info(best)["format"]["bit_rate"])
    standard_rate = int(ffprobe_info(standard)["format"]["bit_rate"])
    assert best_rate > standard_rate, (env, ext, best_rate, standard_rate)


def test_chunk_cache_hits_across_encode_tiers(tmp_path, one_page_pdf, final_voice):
    """Chunk WAVs are identical for standard vs best (same engine/config), so
    the chunk cache keys carry no encode tier; only the final encode differs.
    """
    out_dir = tmp_path / "out"
    ab.convert(
        one_page_pdf,
        out_dir,
        voice=final_voice,
        quality="final",
        encode_tier="standard",
        engine=FakeEngine(),
    )
    fake2 = FakeEngine()
    result = ab.convert(
        one_page_pdf,
        out_dir,
        voice=final_voice,
        quality="final",
        encode_tier="best",
        engine=fake2,
    )
    assert fake2.calls == [], "nothing should be synthesized on a cached re-run"
    assert result.cache_misses == 0
    assert result.cache_hits == result.total_chunks
