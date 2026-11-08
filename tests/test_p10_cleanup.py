"""P10/C1 backend cleanup: delete/cancel must wipe storage.

Offline by construction: the convert subprocess is always a fake Popen (or
never started at all), nothing real is killed, and no engine is constructed.
Style mirrors tests/test_p9_jobs.py.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from audiobookey.cache import book_id as pdf_content_key
from audiobookey.server import db as book_db
from audiobookey.server import jobs as jobs_mod
from audiobookey.server.api import create_app
from audiobookey.server.jobs import JobManager

SPEC = {
    "voice": "af_heart",
    "quality": "draft",
    "gap": 0.3,
    "lexicon_text": "",
    "ocr": False,
    "ocr_lang": "eng",
}


def wait_until(check, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check():
            return True
        time.sleep(interval)
    return False


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)  # keep find_data_dir() off the repo data/
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pdf = tmp_path / "book.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72), "This is page number 1.\nIt has a couple of sentences.", fontsize=12
    )
    doc.save(str(pdf))
    doc.close()
    book_db.init_db(data_dir)
    return {"tmp": tmp_path, "data_dir": data_dir, "pdf": pdf}


def make_client(env, **client_kwargs) -> tuple[TestClient, object]:
    app = create_app(env["data_dir"])
    return TestClient(app, **client_kwargs), app


def plant_book(env, title: str = "kafka", pdf: Path | None = None,
               status: str = "ready") -> tuple[dict, Path]:
    """A book row with every disk artifact it would have after a convert."""
    data_dir = env["data_dir"]
    if pdf is None:
        pdf = data_dir / "uploads" / "kafka-job1.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(env["pdf"].read_bytes())
    book = book_db.add_book(
        title=title,
        pdf_path=str(pdf),
        pages=3,
        status=status,
        data_dir=data_dir,
    )
    book_id = book["id"]
    pages_dir = data_dir / "outputs" / f"book-{book_id}" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page-0001.wav").write_bytes(b"RIFFfake-wav")
    covers = data_dir / "covers"
    covers.mkdir(parents=True, exist_ok=True)
    (covers / f"{book_id}.png").write_bytes(b"\x89PNG-fake-cover")
    key = pdf_content_key(pdf)
    chunks = data_dir / "cache" / key / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    (chunks / "chunk-1.wav").write_bytes(b"RIFFfake-cache")
    # leftover per-book convert tmp for an old dead job (job ids unknown to a
    # delete, so the wipe must find it via the book key)
    old_tmp = data_dir / "convert-tmp" / "job-9" / "cache" / key / "chunks"
    old_tmp.mkdir(parents=True, exist_ok=True)
    (old_tmp / "chunk-2.wav").write_bytes(b"RIFFfake-tmp")
    return book, pdf


def set_spec(env, book_id: int, spec: dict) -> None:
    book_db.set_job_spec(book_id, json.dumps(spec), data_dir=env["data_dir"])


# -- delete: full disk reclaim ---------------------------------------------------


def test_delete_reclaims_all_storage_and_cascades(env) -> None:
    book, pdf = plant_book(env)
    book_id = book["id"]

    # dependent rows must exist first so the cascade has something to do
    book_db.set_progress(book_id, 42.0, 2, data_dir=env["data_dir"])
    book_db.add_bookmark(book_id, 7.5, 1, "memo", data_dir=env["data_dir"])

    data = env["data_dir"]
    key = pdf_content_key(pdf)
    client, app = make_client(env)
    with client:
        resp = client.delete(f"/api/books/{book_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["id"] == book_id

        wiped = "\n".join(body["wiped"])
        # row gone
        assert book_db.get_book(book_id, data_dir=data) is None
        # children gone (ON DELETE CASCADE; PRAGMA foreign_keys = ON in connect)
        conn = sqlite3.connect(data / "audiobookey.db")
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM progress WHERE book_id = ?", (book_id,)
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM bookmarks WHERE book_id = ?", (book_id,)
            ).fetchone()[0] == 0
        finally:
            conn.close()
        # disk artifacts gone: outputs, cover, uploaded pdf, cache, convert tmp
        assert not (data / "outputs" / f"book-{book_id}").exists()
        assert not (data / "covers" / f"{book_id}.png").exists()
        assert not pdf.is_file()
        assert not (data / "cache" / key).exists()
        assert not (data / "convert-tmp" / "job-9" / "cache" / key).exists()
        # the wiped list names every reclaimed path
        assert f"book-{book_id}" in wiped
        assert f"{book_id}.png" in wiped
        assert key in wiped
        assert str(pdf) in wiped

    # deleting again is a clean 404
    client, app = make_client(env)
    with client:
        assert client.delete(f"/api/books/{book_id}").status_code == 404


def test_delete_path_safety_keeps_outside_pdf(env) -> None:
    data = env["data_dir"]
    outside = env["tmp"] / "outside.pdf"  # lives OUTSIDE the data dir
    outside.write_bytes(env["pdf"].read_bytes())
    book, pdf = plant_book(env, pdf=outside)
    key = pdf_content_key(pdf)

    client, app = make_client(env)
    with client:
        resp = client.delete(f"/api/books/{book['id']}")
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True
        # user data outside the data dir is never touched
        assert outside.is_file()
        assert str(outside) not in "\n".join(resp.json()["wiped"])
        # but the book's own artifacts inside the data dir are still wiped
        assert not (data / "outputs" / f"book-{book['id']}").exists()
        assert not (data / "cache" / key).exists()

    assert book_db.get_book(book["id"], data_dir=data) is None


# -- cancel: partial outputs + cache gone, row cancelled, resumable false ---------


class BlockingPopen:
    """Fake convert: the job's first page is already on disk, then it blocks
    until the (faked) tree-kill releases it; it dies non-zero like a killed
    subprocess would."""

    book_id = 1
    job_id = 1
    data_dir: Path
    started: threading.Event
    release: threading.Event

    def __init__(self, cmd, **kwargs):
        self.pid = 4242
        self.returncode = 1
        self.stdout = self._lines()

    @classmethod
    def publish_first_page(cls) -> None:
        pages = cls.data_dir / "outputs" / f"book-{cls.book_id}" / "pages"
        pages.mkdir(parents=True, exist_ok=True)
        (pages / "page-0001.wav").write_bytes(b"RIFFfake-page")

    def _lines(self) -> Iterator[str]:
        self.publish_first_page()
        self.started.set()
        # long enough to survive slow test-client startup; the faked tree-kill
        # (or cancel/pause) releases it, a 10 s expiry made the fake die early
        # and race the pause/cancel request
        self.release.wait(120.0)
        return iter(())
        yield  # pragma: no cover

    def wait(self) -> int:
        self.returncode = 1
        return 1


def fresh_ids(env):
    """A brand-new app has exactly one queued job on book 1."""
    return 1, 1


def test_cancel_wipes_partial_outputs_and_cache(env, monkeypatch) -> None:
    BlockingPopen.data_dir = env["data_dir"]
    BlockingPopen.started = threading.Event()
    BlockingPopen.release = threading.Event()
    monkeypatch.setattr(jobs_mod.subprocess, "Popen", BlockingPopen)

    data = env["data_dir"]
    client, app = make_client(env)
    with client:
        mgr = app.state.jobs
        monkeypatch.setattr(
            mgr, "_kill_process_tree", lambda pid: BlockingPopen.release.set()
        )

        resp = client.post(
            "/api/convert",
            files={"pdf": ("book.pdf", env["pdf"].read_bytes(), "application/pdf")},
            data={"voice": "af_heart", "quality": "draft"},
        )
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]
        book_id = resp.json()["book_ids"][0]
        BlockingPopen.book_id = book_id
        BlockingPopen.job_id = job_id
        assert wait_until(
            lambda: (data / "outputs" / f"book-{book_id}" / "pages" / "page-0001.wav").is_file()
        ), "the fake convert never published its first page"

        # the real pipeline parks chunks under data/cache/<content-key>/chunks
        key = pdf_content_key(data / "uploads" / f"book-job{job_id}.pdf")
        cache_chunk = data / "cache" / key / "chunks" / "chunk-1.wav"
        cache_chunk.parent.mkdir(parents=True, exist_ok=True)
        cache_chunk.write_bytes(b"RIFFfake-cache")
        convert_tmp = data / "convert-tmp" / f"job-{job_id}"
        assert wait_until(lambda: convert_tmp.is_dir())

        resp = client.post(f"/api/books/{book_id}/cancel")
        assert resp.status_code == 202, resp.text

        job = mgr.jobs[job_id]
        # the wipe runs just after the status flip: wait for BOTH the status
        # and the wipe log line, then the disk assertions below are stable
        assert wait_until(
            lambda: job["status"] == "cancelled"
            and any("wiped" in line for line in job["log"])
        ), job["log"]
        # partial outputs, cache dir and convert tmp are gone, immediately
        assert not (data / "outputs" / f"book-{book_id}").exists()
        assert not (data / "cache" / key).exists()
        assert not convert_tmp.exists()
        # the wipe logged a line to the job log
        assert any("wiped" in line for line in job["log"])

        row = book_db.get_book(book_id, data_dir=data)
        assert row is not None
        assert row["status"] == "cancelled"
        assert row["resumable"] is False


def test_cancel_queued_job_wipes_too(env, monkeypatch) -> None:
    """A job cancelled while still queued (tag-and-skip) never starts a
    subprocess — and still reclaims the book's partial output from an
    earlier run on the same row."""
    data = env["data_dir"]
    BlockingPopen.data_dir = data
    BlockingPopen.started = threading.Event()
    BlockingPopen.release = threading.Event()

    class NoSpawnPopen:
        def __init__(self, cmd, **kwargs):
            raise AssertionError("a tagged job must not start a subprocess")

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", NoSpawnPopen)
    client, app = make_client(env)
    with client:
        mgr = app.state.jobs
        monkeypatch.setattr(mgr, "_ensure_worker", lambda: None)

        resp = client.post(
            "/api/convert",
            files={"pdf": ("book.pdf", env["pdf"].read_bytes(), "application/pdf")},
            data={"voice": "af_heart", "quality": "draft"},
        )
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]
        book_id = resp.json()["book_ids"][0]

        # an earlier run on this row left partial output behind
        BlockingPopen.book_id = book_id
        BlockingPopen.publish_first_page()
        key = pdf_content_key(data / "uploads" / f"book-job{job_id}.pdf")
        cache_chunk = data / "cache" / key / "chunks" / "chunk-1.wav"
        cache_chunk.parent.mkdir(parents=True, exist_ok=True)
        cache_chunk.write_bytes(b"RIFFfake-cache")

        resp = client.post(f"/api/books/{book_id}/cancel")
        assert resp.status_code == 202, resp.text

        # the worker skips the tagged task and wipes what the earlier run left
        job = mgr.jobs[job_id]
        task = mgr._queue.get_nowait()
        mgr._run_job(task)  # synchronous: what the drained worker would do
        assert job["status"] == "cancelled"
        assert any("skipped" in line for line in job["log"])
        assert any("wiped" in line for line in job["log"])
        assert not (data / "outputs" / f"book-{book_id}").exists()
        assert not (data / "cache" / key).exists()
        row = book_db.get_book(book_id, data_dir=data)
        assert row is not None
        assert row["status"] == "cancelled"
        assert row["resumable"] is False


# -- pause is the resume path: everything stays -----------------------------------


def test_pause_keeps_partial_pages_and_cache(env, monkeypatch) -> None:
    BlockingPopen.data_dir = env["data_dir"]
    BlockingPopen.started = threading.Event()
    BlockingPopen.release = threading.Event()
    monkeypatch.setattr(jobs_mod.subprocess, "Popen", BlockingPopen)

    data = env["data_dir"]
    client, app = make_client(env)
    with client:
        mgr = app.state.jobs
        monkeypatch.setattr(
            mgr, "_kill_process_tree", lambda pid: BlockingPopen.release.set()
        )

        resp = client.post(
            "/api/convert",
            files={"pdf": ("book.pdf", env["pdf"].read_bytes(), "application/pdf")},
            data={"voice": "af_heart", "quality": "draft"},
        )
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]
        book_id = resp.json()["book_ids"][0]
        BlockingPopen.book_id = book_id
        BlockingPopen.job_id = job_id

        assert wait_until(
            lambda: (data / "outputs" / f"book-{book_id}" / "pages" / "page-0001.wav").is_file()
        )
        key = pdf_content_key(data / "uploads" / f"book-job{job_id}.pdf")
        cache_chunk = data / "cache" / key / "chunks" / "chunk-1.wav"
        cache_chunk.parent.mkdir(parents=True, exist_ok=True)
        cache_chunk.write_bytes(b"RIFFfake-cache")

        resp = client.post(f"/api/books/{book_id}/pause")
        assert resp.status_code == 202, resp.text

        job = mgr.jobs[job_id]
        assert wait_until(lambda: job["status"] == "paused"), job["log"]
        # the row status write trails the in-memory flip by a moment
        row = book_db.get_book(book_id, data_dir=data)
        assert row is not None
        assert wait_until(
            lambda: book_db.get_book(book_id, data_dir=data)["status"] == "paused"
        ), row
        # PAUSE keeps everything: partial pages, chunk cache and convert tmp
        assert (data / "outputs" / f"book-{book_id}" / "pages" / "page-0001.wav").is_file()
        assert cache_chunk.is_file()
        assert (data / "convert-tmp" / f"job-{job_id}").exists()

        row = book_db.get_book(book_id, data_dir=data)
        assert row is not None
        assert row["status"] == "paused"
        assert row["resumable"] is True


def test_requeue_after_cancel_starts_fresh(env, monkeypatch) -> None:
    """A cancelled book requeues fine and starts a FRESH synthesis: its cache
    dir was wiped, so adopt_orphan_caches must find nothing and never crash."""
    env["data_dir"].mkdir(exist_ok=True)
    book, pdf = plant_book(env, status="cancelled")
    set_spec(env, book["id"], SPEC)

    # simulate the cancelled state the user decision produces: no outputs, no
    # cover, no cache, no convert tmp — they were wiped by the cancel
    data = env["data_dir"]
    key = pdf_content_key(pdf)
    shutil.rmtree(data / "outputs" / f"book-{book['id']}")
    (data / "covers" / f"{book['id']}.png").unlink()
    shutil.rmtree(data / "cache" / key)
    shutil.rmtree(data / "convert-tmp")

    mgr = JobManager(data)
    monkeypatch.setattr(mgr, "_ensure_worker", lambda: None)  # never convert

    job = mgr.requeue(book["id"], {})
    assert job["status"] == "queued"
    assert job["message"] == "requeued"  # no "adopted N cached chunks"
    assert book_db.get_book(book["id"], data_dir=data)["status"] == "preparing"


# -- upload tmp leak ----------------------------------------------------------------


def test_convert_upload_tmp_cleaned_on_success(env, monkeypatch) -> None:
    client, app = make_client(env)

    def fake_submit(pdf, **kwargs):
        assert Path(pdf).is_file()
        return {"id": 1, "status": "queued", "book_ids": [1]}

    monkeypatch.setattr(app.state.jobs, "submit", fake_submit)
    with client:
        resp = client.post(
            "/api/convert",
            files={"pdf": ("book.pdf", env["pdf"].read_bytes(), "application/pdf")},
            data={"voice": "af_heart", "quality": "draft"},
        )
        assert resp.status_code == 201, resp.text
        # scratch upload temp removed even though the route "succeeded"
        assert list(env["data_dir"].glob("tmp*.pdf")) == []


def test_convert_upload_tmp_cleaned_on_failure(env, monkeypatch) -> None:
    client, app = make_client(env, raise_server_exceptions=False)

    def boom(*args, **kwargs):
        raise RuntimeError("submit exploded")

    monkeypatch.setattr(app.state.jobs, "submit", boom)
    with client:
        resp = client.post(
            "/api/convert",
            files={"pdf": ("book.pdf", env["pdf"].read_bytes(), "application/pdf")},
            data={"voice": "af_heart", "quality": "draft"},
        )
        assert resp.status_code == 500
        assert list(env["data_dir"].glob("tmp*.pdf")) == []


def test_convert_batch_failure_removes_created_uploads(env, monkeypatch) -> None:
    client, app = make_client(env, raise_server_exceptions=False)

    def boom(*args, **kwargs):
        raise RuntimeError("submit exploded")

    monkeypatch.setattr(app.state.jobs, "submit", boom)
    payload = env["pdf"].read_bytes()
    with client:
        resp = client.post(
            "/api/convert/batch",
            files=[
                ("pdf", ("a.pdf", payload, "application/pdf")),
                ("pdf", ("b.pdf", payload, "application/pdf")),
            ],
            data={"voice": "af_heart", "quality": "draft"},
        )
        assert resp.status_code == 500
        # the two upload copies this request created are gone again
        assert list((env["data_dir"] / "uploads").glob("*.pdf")) == []
