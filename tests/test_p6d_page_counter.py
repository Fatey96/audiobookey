"""P6 regression: the job's "Pages" counter must count PAGES, not chunks.

The convert reporter emits one `update` event per completed page, but that
event's `done` payload is the cumulative CHUNK count. The job manager used to
store it verbatim into page_done, so any page yielding multiple chunks made
the frontend show pages > page_count (e.g. 1060 / 442 on a real novel).
"""

from __future__ import annotations

import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from audiobookey.server.api import create_app


def make_multi_sentence_pdf(path: Path, pages: int = 2) -> Path:
    """Several sentences per page so chunk_total exceeds page_count."""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        # insert_textbox wraps long text across the page; a single
        # insert_text call loses everything past the first visual lines
        text = (
            f"Page {i + 1} begins here. "
            + " ".join(
                f"Entry {n} kept its steady song through the trees."
                for n in range(1, 41)
            )
            + " Night settled in at last."
        )
        page.insert_textbox(
            pymupdf.Rect(72, 72, 523, 770), text, fontsize=12
        )
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def server_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(data_dir))
    monkeypatch.chdir(tmp_path)  # keep find_data_dir() away from the repo data/
    pdf = make_multi_sentence_pdf(tmp_path / "book.pdf", pages=2)
    return {"data_dir": data_dir, "pdf": pdf}


def wait_job_done(client: TestClient, job_id: int, timeout: int = 120) -> dict:
    deadline = time.time() + timeout
    job = client.get(f"/api/jobs/{job_id}").json()
    while time.time() < deadline:
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.4)
        job = client.get(f"/api/jobs/{job_id}").json()
    return job


def test_pages_counter_never_exceeds_page_count(server_env):
    with TestClient(create_app()) as client:
        with open(server_env["pdf"], "rb") as f:
            resp = client.post(
                "/api/convert",
                files={"pdf": ("book.pdf", f, "application/pdf")},
                data={"voice": "af_heart", "quality": "draft", "gap": "0.2"},
            )
        assert resp.status_code == 201, resp.text
        job = wait_job_done(client, resp.json()["id"])
        assert job["status"] == "done", job["message"]

        assert job["page_count"] == 2
        # the book has several sentences per page, so there are more chunks
        # than pages — that is exactly the shape that used to overflow
        assert job["chunk_total"] > job["page_count"]
        assert job["page_done"] == job["page_count"], (job["page_done"], job)
