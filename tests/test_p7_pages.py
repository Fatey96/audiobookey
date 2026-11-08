"""P7-A: live page publishing + book status on the server side.

The pipeline (P7-B) emits ``page_ready``; the job worker must turn each event
into a published WAV + ``pages/index.json`` entry, the book row must exist in
``preparing`` before conversion runs, and the API must serve per-page audio.
No subprocess, no GPU: events are synthetic, end-to-end checks use the fake
engine in-process (same pattern as tests/test_p7_pipeline.py).
"""

from __future__ import annotations

import json
import sqlite3
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from audiobookey.server import db as book_db
from audiobookey.server.api import create_app
from audiobookey.server.jobs import JobManager


SAMPLE_RATE = 24000


# -- helpers -------------------------------------------------------------------


def write_tiny_wav(path: Path, seconds: float = 0.5) -> Path:
    """A real WAV file, small enough to keep the suite fast."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    samples = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    import struct

    data = b"".join(struct.pack("<h", int(s * 32767)) for s in samples)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data)
    return path


def make_pdf(path: Path, pages: int = 2) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        text = (
            f"Page {i + 1} begins here. "
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


def make_manager(tmp_path: Path) -> JobManager:
    return JobManager(tmp_path / "data")


def make_job(book_ids: list[int]) -> dict:
    return {
        "id": 1,
        "status": "running",
        "book_id": book_ids[0] if book_ids else None,
        "message": "",
        "page_count": 0,
        "page_done": 0,
        "chunk_done": 0,
        "chunk_total": 0,
        "cache_hits": 0,
        "files": ["book.pdf"],
        "book_ids": list(book_ids),
        "log": [],
    }


@pytest.fixture()
def server_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(data_dir))
    monkeypatch.chdir(tmp_path)  # keep find_data_dir() away from the repo data/
    book_db.init_db(data_dir)
    return {"data_dir": data_dir, "pdf": make_pdf(tmp_path / "book.pdf", pages=2)}


# 1. counter: one page_ready per page, update events are cache stats only


def test_synthetic_events_count_pages_and_chunks(server_env):
    mgr = make_manager(Path(server_env["data_dir"]).parent)
    book = book_db.add_book(title="b", status="preparing", data_dir=Path(server_env["data_dir"]))
    job = make_job([book["id"]])

    mgr._apply_event(job, {"start_chunks": {"page_count": 2, "chunk_total": 9}},
                     "", 1, 1, "book.pdf", 0, book["id"])
    for n in range(1, 10):
        mgr._apply_event(job, {"chunk_done": n}, "", 1, 1, "book.pdf", 0, book["id"])
    # cache-hit events must not count as finished pages
    mgr._apply_event(job, {"update": {"cache_hits": 4, "done": 4, "total": 9}},
                     "", 1, 1, "book.pdf", 0, book["id"])
    assert job["page_done"] == 0
    assert job["cache_hits"] == 4

    for page, wav in ((1, write_tiny_wav(Path(server_env["data_dir"]) / "p1.wav")),
                      (2, write_tiny_wav(Path(server_env["data_dir"]) / "p2.wav"))):
        mgr._apply_event(
            job,
            {"page_ready": {"page": page, "index": page, "total_pages": 2,
                            "wav": str(wav), "duration_s": 0.5, "sentences": []}},
            "", 1, 1, "book.pdf", 0, book["id"],
        )

    assert job["page_count"] == 2
    assert job["chunk_total"] == 9
    assert job["page_done"] == 2
    assert job["cache_hits"] == 4  # still untouched by page_ready events


# 2. page_ready publishes the wav and builds pages/index.json


def test_page_ready_publishes_wav_and_index(server_env):
    data_dir = Path(server_env["data_dir"])
    mgr = make_manager(data_dir.parent)
    book = book_db.add_book(title="b", pages=2, status="preparing", data_dir=data_dir)
    job = make_job([book["id"]])

    sentences = [{"start_s": 0.0, "end_s": 1.2, "text": "One."}, {"start_s": 1.2, "end_s": 2.4, "text": "Two."}]
    w1 = write_tiny_wav(data_dir / "tmp" / "page-1.wav", seconds=0.6)
    w2 = write_tiny_wav(data_dir / "tmp" / "page-2.wav", seconds=0.3)

    for page, wav in ((1, w1), (2, w2)):
        mgr._apply_event(
            job,
            {"page_ready": {"page": page, "index": page, "total_pages": 2,
                            "wav": str(wav), "duration_s": 0.5 + page / 10.0,
                            "sentences": sentences}},
            "", 1, 1, "book.pdf", 0, book["id"],
        )

    pages_dir = data_dir / "outputs" / f"book-{book['id']}" / "pages"
    index = json.loads((pages_dir / "index.json").read_text(encoding="utf-8"))
    assert index["book_id"] == book["id"]
    assert index["status"] == "preparing"
    assert index["total_pages"] == 2
    assert [p["page"] for p in index["pages"]] == [1, 2]  # ascending book order
    assert all((pages_dir / p["file"]).is_file() for p in index["pages"])
    assert (pages_dir / "page-0001.wav").read_bytes() == w1.read_bytes()
    assert index["pages"][0]["duration_s"] == pytest.approx(0.6)
    assert index["pages"][0]["sentences"] == sentences

    # republishing the same page replaces the entry instead of duplicating it
    mgr._apply_event(
        job,
        {"page_ready": {"page": 1, "index": 1, "total_pages": 2,
                        "wav": str(w1), "duration_s": 9.9, "sentences": []}},
        "", 1, 1, "book.pdf", 0, book["id"],
    )
    index = json.loads((pages_dir / "index.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in index["pages"]] == [1, 2]
    assert index["pages"][0]["duration_s"] == pytest.approx(9.9)


# 3. HTTP: /api/books/{id}/pages and per-page audio


def test_pages_routes(server_env):
    data_dir = Path(server_env["data_dir"])
    mgr = make_manager(data_dir.parent)
    book = book_db.add_book(title="b", pages=3, status="preparing", data_dir=data_dir)
    job = make_job([book["id"]])

    w1 = write_tiny_wav(data_dir / "convert-tmp" / "job-1" / "page-0001.wav")
    w2 = write_tiny_wav(data_dir / "convert-tmp" / "job-1" / "page-0002.wav")
    page_sentences = [
        {"start_s": 0.0, "end_s": 0.2, "text": "One."},
        {"start_s": 0.2, "end_s": 0.5, "text": "Two."},
    ]
    for page, wav in ((1, w1), (2, w2)):
        mgr._apply_event(
            job,
            {"page_ready": {"page": page, "index": page, "total_pages": 3,
                            "wav": str(wav), "duration_s": 0.5 * page,
                            "sentences": page_sentences if page == 1 else []}},
            "", 1, 1, "book.pdf", 0, book["id"],
        )

    with TestClient(create_app()) as client:
        listing = client.get(f"/api/books/{book['id']}/pages").json()
        assert listing["book_id"] == book["id"]
        assert listing["status"] == "preparing"
        assert listing["total_pages"] == 3  # db pages wins while > 0
        assert listing["ready_count"] == 2
        assert [p["page"] for p in listing["pages"]] == [1, 2]
        assert listing["pages"][0]["url"] == f"/api/books/{book['id']}/pages/1"
        assert listing["duration_ready_s"] == pytest.approx(1.5)
        # sentences ride along so live-mode captions can render before the
        # book ever finishes (they are page-local, start_s from 0)
        assert listing["pages"][0]["sentences"] == page_sentences
        assert listing["pages"][1]["sentences"] == []

        body = w1.read_bytes()
        first = client.get(
            f"/api/books/{book['id']}/pages/1", headers={"Range": "bytes=0-1023"}
        )
        assert first.status_code == 206
        assert first.headers["content-range"] == (
            f"bytes 0-1023/{len(body)}"
        )
        assert first.headers["accept-ranges"] == "bytes"
        assert first.content == body[:1024]

        assert client.get(f"/api/books/{book['id']}/pages/99999").status_code == 404
        assert client.get(f"/api/books/{book['id']}/pages/3").status_code == 404


def test_pages_of_unknown_book_is_404(server_env):
    with TestClient(create_app()) as client:
        resp = client.get("/api/books/12345/pages")
        assert resp.status_code == 404
        assert resp.json() == {"error": "book not found"}


def test_pages_without_index_lists_nothing(server_env):
    data_dir = Path(server_env["data_dir"])
    # Book is created while the server is already up: a 'preparing' row that
    # predates boot is legitimately swept to 'error' (see test below).
    with TestClient(create_app()) as client:
        book = book_db.add_book(
            title="b", pages=0, status="preparing", data_dir=data_dir
        )
        listing = client.get(f"/api/books/{book['id']}/pages").json()
        assert listing["ready_count"] == 0
        assert listing["pages"] == []
        assert listing["status"] == "preparing"
        assert listing["total_pages"] == 0


# 4. submit pre-creates the book row; /api/books shows 'preparing'


def test_submit_creates_preparing_book_before_convert(server_env, monkeypatch):
    # App (and its startup sweep) must exist before submit — like the real
    # server, which boots once and then accepts conversions.
    client_cm = TestClient(create_app())
    with client_cm as client:
        def no_run(self, task):  # never convert anything
            task["job"]["status"] = "cancelled"

        monkeypatch.setattr(JobManager, "_run_job", no_run)
        data_dir = Path(server_env["data_dir"])
        mgr = JobManager(data_dir)
        job = mgr.submit(server_env["pdf"], voice="draft", quality="draft", gap=0.3)

        assert job["book_ids"], "submit must pre-create the book row(s)"
        assert len(job["book_ids"]) == 1
        books = client.get("/api/books").json()
        assert len(books) == 1
        assert books[0]["id"] == job["book_ids"][0]
        assert books[0]["status"] == "preparing"
        assert books[0]["ready_pages"] == 0
        assert books[0]["duration_ready_s"] == 0.0


# 5. legacy db migration: status column appears, twice in a row


def test_status_column_migrates_legacy_db(server_env, tmp_path):
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy = legacy_dir / "audiobookey.db"
    conn = sqlite3.connect(legacy)
    conn.execute(
        "CREATE TABLE books (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT,"
        " pdf_path TEXT, m4b_path TEXT, mp3_path TEXT, pages INTEGER,"
        " duration_s REAL)"
    )
    conn.execute("INSERT INTO books (title, pages) VALUES ('old', 5)")
    conn.commit()
    conn.close()

    import audiobookey.server.db as db

    # init_db on the legacy data dir; must be safe twice in a row
    db.init_db(legacy_dir)
    db.init_db(legacy_dir)

    conn = sqlite3.connect(legacy)
    rows = conn.execute("PRAGMA table_info(books)").fetchall()
    names = [r[1] for r in rows]
    status_defs = [r for r in rows if r[1] == "status"]
    legacy_rows = conn.execute("SELECT title, status FROM books").fetchall()
    conn.close()

    assert names.count("status") == 1
    assert status_defs[0][3] == 1  # NOT NULL
    assert "ready" in (status_defs[0][4] or "")
    assert legacy_rows == [("old", "ready")]


# 6. DELETE removes the book row AND its outputs directory


def test_delete_book_removes_outputs_dir(server_env):
    data_dir = Path(server_env["data_dir"])
    mgr = make_manager(data_dir.parent)
    book = book_db.add_book(title="b", pages=2, status="preparing", data_dir=data_dir)
    job = make_job([book["id"]])
    wav = write_tiny_wav(data_dir / "convert-tmp" / "job-1" / "page-0001.wav")
    mgr._apply_event(
        job,
        {"page_ready": {"page": 1, "index": 1, "total_pages": 2,
                        "wav": str(wav), "duration_s": 0.5, "sentences": []}},
        "", 1, 1, "book.pdf", 0, book["id"],
    )
    outputs = data_dir / "outputs" / f"book-{book['id']}"
    assert outputs.is_dir()

    with TestClient(create_app()) as client:
        resp = client.delete(f"/api/books/{book['id']}")
        assert resp.status_code == 200
        # P10 delete also purges uploads/covers/cache and names what it wiped
        assert resp.json()["ok"] is True
        assert resp.json()["id"] == book["id"]
        assert client.get("/api/books").json() == []
        assert not outputs.exists()


# end-to-end: convert page_ready events flow into the worker (fake engine)


def test_convert_page_ready_flows_through_worker(server_env, tmp_path):
    """Run the real in-process convert with the fake engine and feed its
    page_ready events into the worker (subprocess convert would need the
    GPU engine; the LineReporter produces exactly the same payloads)."""
    from audiobookey import audiobook as ab
    from audiobookey.reporter import QuietReporter

    from .fake_engine import FakeEngine

    collected: list[dict] = []

    class Line(QuietReporter):
        def page_ready(self, page, index, total_pages, wav, duration_s, sentences):
            collected.append(
                {"page_ready": {"page": page, "index": index,
                                "total_pages": total_pages, "wav": str(wav),
                                "duration_s": duration_s,
                                "sentences": [dict(s) for s in sentences]}}
            )

    pdf = make_pdf(tmp_path / "book3.pdf", pages=3)
    out = tmp_path / "out"
    result = ab.convert(pdf, out, engine=FakeEngine(), reporter=Line())
    if not collected:
        pytest.skip("P7-B has not landed page_ready yet")
    assert len(collected) == result.chapters == 3

    data_dir = Path(server_env["data_dir"])
    mgr = make_manager(data_dir.parent)
    book = book_db.add_book(title="book3", pages=3, status="preparing", data_dir=data_dir)
    job = make_job([book["id"]])
    for payload in collected:
        mgr._apply_event(job, payload, "", 1, 1, "book3.pdf", 0, book["id"])

    assert job["page_done"] == 3
    pages_dir = data_dir / "outputs" / f"book-{book['id']}" / "pages"
    index = json.loads((pages_dir / "index.json").read_text(encoding="utf-8"))
    assert [p["page"] for p in index["pages"]] == [1, 2, 3]
    assert all((pages_dir / p["file"]).is_file() for p in index["pages"])
    assert index["pages"][0]["sentences"], "page-local sentences survive the hop"
