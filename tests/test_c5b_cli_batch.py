"""C5B tests: one engine across the CLI multi-PDF batch + explicit VRAM release.

The convert loop used to call convert() per PDF with engine=None, so every
book paid a fresh model load + canary synthesis. These tests pin the new
behaviour without any real synthesis, model load, or GPU:

- one engine construction for the whole batch, and convert() receives that
  same engine object for every book;
- after the batch, the CUDA cache is emptied exactly once (fake torch);
- a torch-less (CPU-only) install still completes the batch;
- a failed book keeps today's failure semantics (error surfaces, run aborts
  with rc 2) and the VRAM release still runs.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pymupdf

from audiobookey import cli as cli_mod
from audiobookey import engine_pool

from .fake_engine import FakeEngine


def _one_page_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class _FakeResult:
    """Just enough of ConvertResult for cli main()'s result printing."""

    def __init__(self, out_dir, name: str) -> None:
        out_dir = Path(out_dir)
        self.m4b = out_dir / f"{name}.m4b"
        self.mp3 = out_dir / f"{name}.mp3"
        self.chapters = 1
        self.duration_seconds = 12.0
        self.skipped_pages: list[int] = []
        self.cache_hits = 0
        self.cache_misses = 2
        self.cache_hit_rate = 0.0


def _patch_env(tmp_path, monkeypatch) -> list[tuple[str, FakeEngine]]:
    """Stub the engine factory (records constructions) + models dir."""
    engines: list[tuple[str, FakeEngine]] = []

    def fake_make_engine(quality, models_dir):
        made = FakeEngine()
        engines.append((quality, made))
        return made

    monkeypatch.setattr(engine_pool, "make_engine", fake_make_engine)
    monkeypatch.setattr(
        cli_mod, "find_models_dir", lambda _models_dir=None: tmp_path / "models"
    )
    return engines


def _fake_torch(empty_cache_calls: list[int]) -> types.ModuleType:
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: empty_cache_calls.append(1),
    )
    return fake


def test_cli_batch_builds_one_engine_for_both_books(tmp_path, monkeypatch):
    pdf1 = _one_page_pdf(
        tmp_path / "book1.pdf",
        "The first book has a page of words. The page simply ends here.",
    )
    pdf2 = _one_page_pdf(
        tmp_path / "book2.pdf",
        "The second book also has a page of words. That page ends too.",
    )
    out = tmp_path / "out"

    engines = _patch_env(tmp_path, monkeypatch)

    real_convert = cli_mod.convert
    passed_engines: list = []

    def recording_convert(*args, **kwargs):
        passed_engines.append(kwargs["engine"])
        return real_convert(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "convert", recording_convert)

    rc = cli_mod.main(
        ["convert", str(pdf1), str(pdf2), "--out", str(out), "--quality", "low"]
    )

    assert rc == 0
    # The engine kind resolves through QUALITY_ENGINE_ALIASES (low -> draft)
    # and the factory runs exactly once for the whole batch...
    assert len(engines) == 1
    assert engines[0][0] == "draft"
    # ...and both books receive that same engine object.
    assert passed_engines == [engines[0][1], engines[0][1]]
    # Both books really converted (real convert() ran with the fake engine).
    assert (out / "book1.m4b").is_file()
    assert (out / "book2.m4b").is_file()


def test_cli_batch_releases_vram_after_loop(tmp_path, monkeypatch):
    pdf1 = _one_page_pdf(tmp_path / "b1.pdf", "Some words for the page here.")
    pdf2 = _one_page_pdf(tmp_path / "b2.pdf", "Other words for another page.")

    _patch_env(tmp_path, monkeypatch)

    empty_cache_calls: list[int] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(empty_cache_calls))

    def stub_convert(*args, **kwargs):
        return _FakeResult(args[1], "book")

    monkeypatch.setattr(cli_mod, "convert", stub_convert)

    rc = cli_mod.main(["convert", str(pdf1), str(pdf2), "--out", str(tmp_path / "o")])

    assert rc == 0
    assert empty_cache_calls == [1], "empty_cache must run once, after the batch"


def test_cli_batch_torch_absent_still_succeeds(tmp_path, monkeypatch):
    pdf = _one_page_pdf(tmp_path / "solo.pdf", "A single page of words to convert.")

    _patch_env(tmp_path, monkeypatch)

    def stub_convert(*args, **kwargs):
        return _FakeResult(args[1], "book")

    monkeypatch.setattr(cli_mod, "convert", stub_convert)

    # A None entry in sys.modules makes `import torch` raise ImportError:
    # exactly what a CPU-only install without torch hits.
    monkeypatch.setitem(sys.modules, "torch", None)

    rc = cli_mod.main(["convert", str(pdf), "--out", str(tmp_path / "out")])

    assert rc == 0, "a torch-less install must complete the run cleanly"


def test_cli_batch_failed_book_still_releases_vram(tmp_path, monkeypatch):
    pdf = _one_page_pdf(tmp_path / "bad.pdf", "Words that will fail to convert.")

    _patch_env(tmp_path, monkeypatch)

    empty_cache_calls: list[int] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(empty_cache_calls))

    def exploding_convert(*args, **kwargs):
        raise RuntimeError("synthesis exploded")

    monkeypatch.setattr(cli_mod, "convert", exploding_convert)

    rc = cli_mod.main(["convert", str(pdf), "--out", str(tmp_path / "out")])

    # Failure semantics unchanged: the error surfaces, the run aborts...
    assert rc == 2
    # ...and the VRAM release still ran afterwards.
    assert empty_cache_calls == [1]
