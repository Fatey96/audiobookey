"""P9 job controls: stored job specs, requeue (resume), pause/cancel
kill-and-mark mechanics, and orphan cache adoption.

Offline by construction: the convert subprocess is always a fake Popen (or
never started at all), nothing real is killed, and no engine is constructed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pymupdf
import pytest

from audiobookey.cache import book_id as pdf_content_key
from audiobookey.server import db as book_db
from audiobookey.server import jobs as jobs_mod
from audiobookey.server.jobs import (
    JobConflict,
    JobInvalidParams,
    JobManager,
    JobNotFound,
)

SPEC = {
    "voice": "af_heart",
    "quality": "draft",
    "gap": 0.3,
    "lexicon_text": "",
    "ocr": False,
    "ocr_lang": "eng",
}


def wait_until(check, timeout=10.0, interval=0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check():
            return True
        time.sleep(interval)
    return False


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep resolve_data_dir() off the repo data/
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


def add_book(env, title="kafka", pdf=None, status="error") -> dict:
    return book_db.add_book(
        title=title,
        pdf_path=str(pdf or env["pdf"]),
        pages=10,
        status=status,
        data_dir=env["data_dir"],
    )


def set_spec(env, book_id: int, spec: dict) -> None:
    book_db.set_job_spec(
        book_id, json.dumps(spec), data_dir=env["data_dir"]
    )


def stored_spec(env, book_id: int) -> dict:
    return json.loads(book_db.get_job_spec(book_id, data_dir=env["data_dir"]))


def make_task(mgr: JobManager, job_id: int, book_ids: list[int], pdf: Path,
              status: str = "queued"):
    """A crafted job + queue task without going through submit()."""
    job = {
        "id": job_id,
        "status": status,
        "book_id": book_ids[0],
        "message": "",
        "page_count": 0,
        "page_done": 0,
        "chunk_done": 0,
        "chunk_total": 0,
        "cache_hits": 0,
        "files": [pdf.name],
        "book_ids": list(book_ids),
        "log": [],
    }
    mgr.jobs[job_id] = job
    task = {
        "job": job,
        "pdfs": [pdf],
        "stems": ["book"],
        "voice": "af_heart",
        "quality": "draft",
        "gap": 0.3,
        "lexicon_file": None,
        "ocr": False,
        "ocr_lang": "eng",
        "book_ids": list(book_ids),
    }
    return job, task


def book_status(env, book_id: int) -> str:
    return book_db.get_book(book_id, data_dir=env["data_dir"])["status"]


# -- job_spec migration -------------------------------------------------------


def test_migration_adds_job_spec_and_is_idempotent(tmp_path):
    # a pre-P9 database: books table already had status, but no job_spec
    conn = sqlite3.connect(tmp_path / "audiobookey.db")
    conn.executescript(
        """
        CREATE TABLE books (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            pdf_path    TEXT NOT NULL DEFAULT '',
            m4b_path    TEXT NOT NULL DEFAULT '',
            mp3_path    TEXT NOT NULL DEFAULT '',
            pages       INTEGER NOT NULL DEFAULT 0,
            duration_s  REAL NOT NULL DEFAULT 0.0,
            created_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')),
            status      TEXT NOT NULL DEFAULT 'ready'
        );
        """
    )
    conn.execute(
        "INSERT INTO books (title, pdf_path, status) VALUES ('kafka', '', 'error')"
    )
    conn.commit()
    conn.close()

    book_db.init_db(tmp_path)
    book_db.init_db(tmp_path)  # second boot: no duplicate-column error

    conn = sqlite3.connect(tmp_path / "audiobookey.db")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(books)")}
    conn.close()
    assert {"status", "job_spec"} <= columns

    assert book_db.get_job_spec(1, data_dir=tmp_path) == ""
    book_db.set_job_spec(1, '{"voice": "af_heart"}', data_dir=tmp_path)
    spec = json.loads(book_db.get_job_spec(1, data_dir=tmp_path))
    assert spec == {"voice": "af_heart"}

    book = book_db.get_book(1, data_dir=tmp_path)
    assert book["job_spec"] == {"voice": "af_heart"}
    # error row, but its pdf_path is gone -> not resumable
    assert book["resumable"] is False


# -- row mapping: resumable + job_spec -----------------------------------------


def test_book_row_resumable_flag(env):
    ids = {}
    for status in ("ready", "preparing", "error", "paused", "cancelled"):
        book = add_book(env, title=f"b-{status}", status=status)
        ids[status] = book["id"]

    books = {b["id"]: b for b in book_db.list_books(env["data_dir"])}
    assert books[ids["ready"]]["resumable"] is False
    assert books[ids["preparing"]]["resumable"] is False
    for status in ("error", "paused"):
        assert books[ids[status]]["resumable"] is True
        assert books[ids[status]]["job_spec"] == {}
    # C1: cancelled books are terminal (cancel wipes their disk state) and
    # must not advertise resumable anymore
    assert books[ids["cancelled"]]["resumable"] is False

    # resumable requires the uploaded pdf to still exist
    gone = add_book(
        env, title="gone", pdf=env["data_dir"] / "gone.pdf", status="error"
    )
    book = book_db.get_book(gone["id"], data_dir=env["data_dir"])
    assert book["resumable"] is False


# -- submit persists the job spec -----------------------------------------------


def test_submit_persists_job_spec_on_book_row(env, monkeypatch):
    mgr = JobManager(env["data_dir"])
    monkeypatch.setattr(mgr, "_ensure_worker", lambda: None)  # never convert
    job = mgr.submit(
        env["pdf"],
        voice="af_heart",
        quality="draft",
        gap=0.25,
        lexicon_text="custom words",
        ocr=True,
        ocr_lang="deu",
    )

    books = book_db.list_books(env["data_dir"])
    assert len(books) == 1
    assert books[0]["status"] == "preparing"
    spec = stored_spec(env, books[0]["id"])
    assert spec == {
        "voice": "af_heart",
        "quality": "draft",
        "gap": 0.25,
        "lexicon_text": "custom words",
        "ocr": True,
        "ocr_lang": "deu",
    }
    assert job["book_id"] == books[0]["id"]
    assert mgr._queue.qsize() == 1


# -- requeue: same row, spec merge, error mapping -------------------------------


def test_requeue_reuses_row_and_updates_spec(env, monkeypatch):
    mgr = JobManager(env["data_dir"])
    monkeypatch.setattr(mgr, "_ensure_worker", lambda: None)
    book = add_book(env, status="error")
    book_id = book["id"]
    set_spec(env, book_id, SPEC)

    job = mgr.requeue(book_id, {"quality": "low"})
    assert job["status"] == "queued"
    assert job["book_id"] == book_id
    assert job["book_ids"] == [book_id]
    assert job["message"] == "requeued"

    # same book row: no duplicate id, status back to preparing
    books = book_db.list_books(env["data_dir"])
    assert [b["id"] for b in books] == [book_id]
    assert books[0]["status"] == "preparing"

    # spec merge: body override wins, untouched keys survive
    spec = stored_spec(env, book_id)
    assert spec["quality"] == "low"
    assert spec["voice"] == "af_heart"
    assert spec["gap"] == 0.3

    task = mgr._queue.get_nowait()
    assert task["quality"] == "low"
    assert task["voice"] == "af_heart"
    assert task["pdfs"] == [Path(env["pdf"])]
    assert task["book_ids"] == [book_id]

    # a live job guards the book against a second requeue
    with pytest.raises(JobConflict):
        mgr.requeue(book_id, {})

    # no stored spec + no params -> helpful rejection
    legacy = add_book(env, title="legacy", status="error")
    with pytest.raises(JobInvalidParams, match="parameters"):
        mgr.requeue(legacy["id"], {})

    # a legacy book requeues fine from body params alone (defaults apply)
    job2 = mgr.requeue(legacy["id"], {"voice": "af_bella", "quality": "medium"})
    assert job2["status"] == "queued"
    assert stored_spec(env, legacy["id"]) == {
        "voice": "af_bella",
        "quality": "medium",
        "gap": 0.3,  # default
        "lexicon_text": "",
        "ocr": False,
        "ocr_lang": "eng",
    }

    # bad gap in the body
    other = add_book(env, title="other", status="paused")
    set_spec(env, other["id"], SPEC)
    with pytest.raises(JobInvalidParams, match="gap"):
        mgr.requeue(other["id"], {"gap": "fast"})

    # pdf gone from disk -> cannot requeue
    gone = add_book(
        env, title="gone", pdf=env["data_dir"] / "gone.pdf", status="cancelled"
    )
    set_spec(env, gone["id"], SPEC)
    with pytest.raises(JobInvalidParams, match="pdf"):
        mgr.requeue(gone["id"], {})

    with pytest.raises(JobNotFound):
        mgr.requeue(9999, {"voice": "af_heart", "quality": "draft"})


# -- queued tag-and-skip ---------------------------------------------------------


def test_run_job_skips_tagged_task_without_spawning(env, monkeypatch):
    spawned: list = []

    class NoSpawnPopen:
        def __init__(self, cmd, **kwargs):
            spawned.append(cmd)
            raise AssertionError("a tagged job must not start a subprocess")

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", NoSpawnPopen)
    mgr = JobManager(env["data_dir"])

    for job_id, outcome in ((1001, "cancelled"), (1002, "paused")):
        book = add_book(env, title=f"b-{outcome}", status="preparing")
        job, task = make_task(
            mgr, job_id, [book["id"]], env["pdf"], status=outcome
        )
        mgr._run_job(task)  # synchronous: the queue worker would do this
        assert spawned == []
        assert job["status"] == outcome  # the tag survives, no overwrite
        assert any("skipped" in line for line in job["log"])
        assert book_status(env, book["id"]) == outcome


def test_queued_cancel_tag_and_skip_fifo(env, monkeypatch):
    """Two fake queued tasks: the first runs, the second is cancelled while
    still queued and must be skipped without ever spawning a subprocess."""
    started = threading.Event()
    release = threading.Event()

    class BlockingPopen:
        calls: list = []

        def __init__(self, cmd, **kwargs):
            BlockingPopen.calls.append(cmd)
            self.pid = 4242
            self.returncode = 0
            self.stdout = self._lines()

        def _lines(self):
            started.set()
            release.wait(10.0)
            return
            yield  # pragma: no cover - generator only, EOF after release

        def wait(self):
            self.returncode = 0
            return 0

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", BlockingPopen)
    mgr = JobManager(env["data_dir"])
    book_a = add_book(env, title="first", status="preparing")
    book_b = add_book(env, title="second", status="preparing")

    job_a, task_a = make_task(mgr, 2001, [book_a["id"]], env["pdf"])
    mgr._queue.put(task_a)
    mgr._ensure_worker()
    assert started.wait(10.0), "the worker never spawned the first job"

    job_b, task_b = make_task(mgr, 2002, [book_b["id"]], env["pdf"])
    mgr._queue.put(task_b)

    resp = mgr.cancel(book_b["id"])
    assert resp == {"job_id": 2002, "status": "cancelled"}
    assert job_b["status"] == "cancelled"
    assert book_status(env, book_b["id"]) == "cancelled"

    # let the running job finish (registration fails: no m4b from the fake)
    release.set()
    assert wait_until(lambda: job_a["status"] == "error"), job_a["log"]
    assert wait_until(lambda: mgr._queue.qsize() == 0)
    time.sleep(0.1)  # let the skipped branch land on the dequeued task

    # job B was skipped, never converted; job A owns the only Popen call
    assert len(BlockingPopen.calls) == 1
    assert job_b["status"] == "cancelled"
    assert any("skipped" in line for line in job_b["log"])
    assert book_status(env, book_b["id"]) == "cancelled"
    # bug-fix behaviour on the failed job A: error in-session, not 'preparing'
    assert book_status(env, book_a["id"]) == "error"


# -- running kill-and-mark: outcome mapping --------------------------------------


@pytest.mark.parametrize("action,outcome", [("cancel", "cancelled"), ("pause", "paused")])
def test_running_stop_maps_outcome_not_error(env, monkeypatch, action, outcome):
    started = threading.Event()
    release = threading.Event()

    class KillablePopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 4321
            self.returncode = 1  # a killed convert exits non-zero
            self.stdout = self._lines()

        def _lines(self):
            started.set()
            release.wait(10.0)  # "synthesis" until the fake taskkill lands
            return
            yield  # pragma: no cover

        def wait(self):
            self.returncode = 1
            return 1

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", KillablePopen)
    mgr = JobManager(env["data_dir"])
    killed: list = []

    def fake_kill(pid):
        killed.append(pid)
        release.set()

    monkeypatch.setattr(mgr, "_kill_process_tree", fake_kill)

    book = add_book(env, status="preparing")
    job, task = make_task(mgr, 3001, [book["id"]], env["pdf"])
    mgr._queue.put(task)
    mgr._ensure_worker()
    assert started.wait(10.0), "the worker never spawned the job"
    assert job["status"] == "running"
    assert job["pid"] == 4321

    resp = getattr(mgr, action)(book["id"])
    assert resp == {"job_id": 3001, "status": outcome}
    assert killed == [4321]

    assert wait_until(lambda: job["status"] == outcome), job["log"]
    assert "by user" in job["message"]
    assert outcome in job["message"]
    assert "error" != job["status"]
    assert book_status(env, book["id"]) == outcome


def test_running_stop_without_live_job_conflicts(env):
    mgr = JobManager(env["data_dir"])
    book = add_book(env, status="error")
    with pytest.raises(JobConflict):
        mgr.cancel(book["id"])
    with pytest.raises(JobConflict):
        mgr.pause(book["id"])


# -- convert failure sets the book to error in-session (bug fix) -----------------


def test_convert_failure_sets_book_error_in_session(env, monkeypatch):
    class FailingPopen:
        def __init__(self, cmd, **kwargs):
            self.pid = 555
            self.returncode = 1
            self.stdout = []

        def wait(self):
            self.returncode = 1
            return 1

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", FailingPopen)
    mgr = JobManager(env["data_dir"])
    book = add_book(env, status="preparing")
    job, task = make_task(mgr, 4001, [book["id"]], env["pdf"])

    mgr._run_job(task)  # synchronous, no worker

    assert job["status"] == "error"
    assert "convert exited with code 1" in job["message"]
    # the latent bug used to leave the book 'preparing' until the restart sweep
    assert book_status(env, book["id"]) == "error"


def test_worker_exception_maps_stop_request_outcome(env):
    """A pause/cancel request that lands mid-crash must not mislabel the
    book as error (the worker's exception path honors cancel_requested)."""
    mgr = JobManager(env["data_dir"])
    book = add_book(env, status="preparing")
    job, task = make_task(mgr, 5001, [book["id"]], env["pdf"])
    task.pop("pdfs")  # _run_job raises before/around Popen -> worker except path
    job["status"] = "running"
    job["cancel_requested"] = True
    job["requested_outcome"] = "paused"

    mgr._queue.put(task)
    mgr._ensure_worker()  # daemon thread; drains one task, then idles

    assert wait_until(lambda: job["status"] == "paused"), job["log"]
    assert "paused by user" in job["message"]
    assert book_status(env, book["id"]) == "paused"


def test_cancel_abandons_paused_book_without_live_job(env, monkeypatch):
    """Cancel on a paused book: no live job to kill, so the row abandons
    (paused -> cancelled) and stays requeueable. Non-paused books without a
    live job keep the 409."""
    mgr = JobManager(env["data_dir"])
    monkeypatch.setattr(mgr, "_ensure_worker", lambda: None)
    book = add_book(env, status="paused")
    set_spec(env, book["id"], SPEC)

    out = mgr.cancel(book["id"])

    assert out == {"job_id": None, "status": "cancelled"}
    assert book_status(env, book["id"]) == "cancelled"

    # abandoned stays requeueable
    job = mgr.requeue(book["id"], {})
    assert job["status"] == "queued"

    with pytest.raises(JobConflict):
        mgr.cancel(add_book(env, title="idle", status="error")["id"])


def test_requeue_accepts_paused_book(env, monkeypatch):
    """Paused rows requeue one-click (same path as error/cancelled rows)."""
    mgr = JobManager(env["data_dir"])
    monkeypatch.setattr(mgr, "_ensure_worker", lambda: None)
    book = add_book(env, status="paused")
    set_spec(env, book["id"], SPEC)

    job = mgr.requeue(book["id"], {})

    assert job["status"] == "queued"
    assert book_status(env, book["id"]) == "preparing"


# -- orphan cache adoption --------------------------------------------------------


def test_adopt_orphan_caches_moves_matching_chunks_only(env, monkeypatch):
    key = pdf_content_key(env["pdf"])
    other_key = "00000000000f"
    data = env["data_dir"]

    plain = data / "convert-tmp" / "job-1" / "cache" / key / "chunks"  # job-2 layout
    nested = data / "convert-tmp" / "job-2" / "cache" / "voice-draft" / key / "chunks"
    stranger = data / "convert-tmp" / "job-3" / "cache" / other_key / "chunks"
    for chunks in (plain, nested, stranger):
        chunks.mkdir(parents=True)
    (plain / "a.wav").write_text("AAA")
    (plain / "b.wav").write_text("BBB")
    (nested / "c.wav").write_text("CCC")
    (stranger / "d.wav").write_text("DDD")
    dest = data / "cache" / key / "chunks"
    dest.mkdir(parents=True)
    (dest / "a.wav").write_text("KEEP")  # existing name must not be replaced

    mgr = JobManager(env["data_dir"])
    adopted = mgr.adopt_orphan_caches(env["pdf"])
    assert adopted == 2  # b + c; a already existed at the destination
    assert (dest / "a.wav").read_text() == "KEEP"
    assert (dest / "b.wav").read_text() == "BBB"
    assert (dest / "c.wav").read_text() == "CCC"  # legacy nested layout adopted
    assert (stranger / "d.wav").read_text() == "DDD"  # other book: untouched
    assert (plain / "a.wav").read_text() == "AAA"  # skipped, stays behind

    # adopting again moves nothing new
    assert mgr.adopt_orphan_caches(env["pdf"]) == 0


def test_adopt_all_orphan_caches_startup_sweep(env):
    """The startup sweep adopts EVERY book's orphans, not just the
    requeued book's: job ids restart at 1 with the server, so a targeted
    sweep would leave other books exposed to the next session's
    same-id job wiping convert-tmp/job-<id>/."""
    key = pdf_content_key(env["pdf"])
    other_key = "00000000000f"
    data = env["data_dir"]

    mine = data / "convert-tmp" / "job-1" / "cache" / key / "chunks"
    stranger = (
        data / "convert-tmp" / "job-2" / "cache" / "voice-final" / other_key
        / "chunks"
    )
    for chunks in (mine, stranger):
        chunks.mkdir(parents=True)
    (mine / "k.wav").write_text("K")
    (stranger / "s.wav").write_text("S")

    mgr = JobManager(env["data_dir"])
    moved = mgr.adopt_all_orphan_caches()

    assert moved == {key: 1, other_key: 1}
    assert (data / "cache" / key / "chunks" / "k.wav").read_text() == "K"
    assert (
        data / "cache" / other_key / "chunks" / "s.wav"
    ).read_text() == "S"  # legacy nested layout adopted too

    # idempotent: nothing left to move
    assert mgr.adopt_all_orphan_caches() == {}


def test_requeue_reports_adopted_chunks_in_message(env, monkeypatch):
    key = pdf_content_key(env["pdf"])
    chunks = env["data_dir"] / "convert-tmp" / "job-2" / "cache" / key / "chunks"
    chunks.mkdir(parents=True)
    (chunks / "orphan.wav").write_text("SYNTH")

    mgr = JobManager(env["data_dir"])
    monkeypatch.setattr(mgr, "_ensure_worker", lambda: None)
    book = add_book(env, status="error")
    set_spec(env, book["id"], SPEC)

    job = mgr.requeue(book["id"], {})
    assert "adopted 1" in job["message"]
    dest = env["data_dir"] / "cache" / key / "chunks"
    assert (dest / "orphan.wav").read_text() == "SYNTH"
    assert not (chunks / "orphan.wav").exists()  # moved, not copied
