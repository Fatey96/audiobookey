"""P6-C regression tests: books registered by finished convert jobs must be
listed by GET /api/books.

Uses a mocked convert subprocess (no real engine, no GPU): the fake Popen
drops a tiny .m4b into the job's --out dir and exits 0, so the job walks the
same submit -> worker -> _register_result -> db.add_book code path that the
real pipeline runs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient


def make_text_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return path


class _FakeConvertProc:
    """Stands in for `python -m audiobookey convert ...`."""

    def __init__(self, cmd: list[str], **kwargs: object) -> None:
        out_dir = Path(cmd[cmd.index("--out") + 1])
        work_pdf = Path(cmd[cmd.index("convert") + 1])
        stem = work_pdf.stem
        # the real convert writes ffmpeg containers; registration only needs
        # a file it can hash-copy around
        (out_dir / f"{stem}.m4b").write_bytes(b"0" * 512)
        (out_dir / f"{stem}.mp3").write_bytes(b"0" * 512)
        (out_dir / "manifest.json").write_text(
            json.dumps({"sentences": [{"text": "hi", "start_s": 0.0, "end_s": 1.0}]}),
            encoding="utf-8",
        )
        self.stdout = [f'EVENT {json.dumps({"phase": "fake convert"})}\n']
        self.returncode = 0

    def wait(self) -> int:
        return 0


def wait_job_done(client: TestClient, job_id: int, timeout: int = 30) -> dict:
    deadline = time.time() + timeout
    job = client.get(f"/api/jobs/{job_id}").json()
    while time.time() < deadline:
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
        job = client.get(f"/api/jobs/{job_id}").json()
    return job


@pytest.fixture()
def server_env(tmp_path, monkeypatch):
    """ Mimics the LIVE server entry point: NO AUDIOBOOKEY_DATA_DIR override.
    find_data_dir() then falls back to cwd/data, which must exist before the
    app is built to stay hermetic (otherwise it would find the repo's data/)."""
    monkeypatch.delenv("AUDIOBOOKEY_DATA_DIR", raising=False)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    pdf_a = make_text_pdf(tmp_path / "book-a.pdf", "Book number one text.")
    pdf_b = make_text_pdf(tmp_path / "book-b.pdf", "Book number two text.")
    return {"data_dir": data_dir, "pdf_a": pdf_a, "pdf_b": pdf_b}


@pytest.fixture()
def mock_convert(monkeypatch):
    monkeypatch.setattr(
        "audiobookey.server.jobs.subprocess.Popen", _FakeConvertProc
    )


def make_client():
    from audiobookey.server.api import create_app

    return TestClient(create_app())


def test_single_convert_job_book_is_listed(server_env, mock_convert):
    with make_client() as client:
        with open(server_env["pdf_a"], "rb") as f:
            resp = client.post(
                "/api/convert",
                files={"pdf": ("book-a.pdf", f, "application/pdf")},
                data={"voice": "af_heart", "quality": "draft", "gap": "0.2"},
            )
        assert resp.status_code == 201, resp.text
        job = wait_job_done(client, resp.json()["id"])
        print("job message:", job["message"])
        assert job["status"] == "done", job["log"]

        books = client.get("/api/books").json()
        assert len(books) == 1, "registered book invisible in /api/books"
        assert books[0]["id"] == job["book_id"]
        assert books[0]["title"] == "book-a"
        assert os.path.isfile(books[0]["m4b_path"])


def test_batch_convert_all_books_are_listed(server_env, mock_convert):
    with make_client() as client:
        with open(server_env["pdf_a"], "rb") as fa, open(
            server_env["pdf_b"], "rb"
        ) as fb:
            resp = client.post(
                "/api/convert/batch",
                files=[
                    ("pdf", ("book-a.pdf", fa, "application/pdf")),
                    ("pdf", ("book-b.pdf", fb, "application/pdf")),
                ],
            )
        assert resp.status_code == 201, resp.text
        job = wait_job_done(client, resp.json()["id"])
        print("job message:", job["message"])
        assert job["status"] == "done", job["log"]

        titles = {b["title"] for b in client.get("/api/books").json()}
        assert titles == {"book-a", "book-b"}, "batch lost a registered book"
        # job bookkeeping stays consistent with the library
        assert sorted(job["book_ids"]) == sorted(
            b["id"] for b in client.get("/api/books").json()
        )
        assert job["book_id"] == job["book_ids"][-1]


# -- the data-dir fork that can make a registered book invisible ----------------


def test_resolve_data_dir_explicit_argument_beats_env(server_env, monkeypatch):
    from audiobookey.server import db

    here = server_env["data_dir"]
    there = server_env["pdf_a"].parent / "elsewhere-data"
    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(there))
    assert db.resolve_data_dir(here) == here, (
        "an ambient AUDIOBOOKEY_DATA_DIR must not override an explicit data_dir"
    )


def test_job_registers_into_explicit_data_dir_not_env(server_env, mock_convert, monkeypatch):
    """Regression for the 'book registered but never listed' report: writers
    (JobManager -> db.add_book) and readers (GET /api/books) must never land in
    different database files, whatever the environment says."""
    from audiobookey.server import db

    explicit = server_env["data_dir"] / "explicit-app-data"
    elsewhere = server_env["data_dir"] / "elsewhere-data"
    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(elsewhere))

    from audiobookey.server.api import create_app

    with TestClient(create_app(data_dir=explicit)) as client:
        with open(server_env["pdf_a"], "rb") as f:
            resp = client.post(
                "/api/convert",
                files={"pdf": ("book-a.pdf", f, "application/pdf")},
                data={"voice": "af_heart", "quality": "draft", "gap": "0.2"},
            )
        assert resp.status_code == 201, resp.text
        job = wait_job_done(client, resp.json()["id"])
        print("job message:", job["message"])
        assert job["status"] == "done", job["log"]

        books = client.get("/api/books").json()
        assert len(books) == 1, "book went to the wrong database"
        # the row physically lives in the explicit dir's database
        conn = db.connect(explicit)
        try:
            rows = conn.execute("SELECT id, title FROM books").fetchall()
        finally:
            conn.close()
        assert [tuple(r) for r in rows] and rows[0]["title"] == "book-a"
        assert not db.db_path(elsewhere).exists(), (
            "no second database may appear under the env dir"
        )
