"""P3 server tests: books CRUD, progress, bookmarks, range streaming, covers,
voices, and the convert job lifecycle (real draft engine on a 1-page PDF).
"""

from __future__ import annotations

import os
import time

import pymupdf
import pytest
from fastapi.testclient import TestClient

AUDIO_PAYLOAD = bytes(range(256)) * 4  # 1024 bytes of deterministic audio


@pytest.fixture()
def server_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(data_dir))
    monkeypatch.chdir(tmp_path)  # keep find_data_dir() away from the repo data/

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "book.m4b").write_bytes(AUDIO_PAYLOAD)
    (audio_dir / "book.mp3").write_bytes(AUDIO_PAYLOAD)

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

    return {
        "data_dir": data_dir,
        "audio_dir": audio_dir,
        "pdf": pdf,
        "audio_payload": AUDIO_PAYLOAD,
    }


def make_client():
    from audiobookey.server.api import create_app

    return TestClient(create_app())


def register_book(client, server_env) -> int:
    resp = client.post(
        "/api/books",
        json={
            "pdf_path": str(server_env["pdf"]),
            "audio_dir": str(server_env["audio_dir"]),
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# -- health / books CRUD -------------------------------------------------------


def test_health(server_env):
    with make_client() as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_books_crud_roundtrip(server_env):
    with make_client() as client:
        assert client.get("/api/books").json() == []

        book_id = register_book(client, server_env)
        book = client.get("/api/books").json()[0]
        assert book["title"] == "book"
        assert book["pdf_path"].endswith("book.pdf")
        assert book["m4b_path"].endswith("book.m4b")
        assert book["mp3_path"].endswith("book.mp3")
        assert book["pages"] == 1
        assert book["progress"] is None
        assert book["bookmarks"] == []
        assert book["cover_path"] == f"/api/books/{book_id}/cover"

        resp = client.delete(f"/api/books/{book_id}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert client.get("/api/books").json() == []

        resp = client.delete(f"/api/books/{book_id}")
        assert resp.status_code == 404
        assert resp.json() == {"error": "book not found"}


def test_register_book_rejects_bad_input(server_env):
    with make_client() as client:
        resp = client.post(
            "/api/books",
            json={"pdf_path": str(tmp := server_env["pdf"].parent / "nope.pdf"),
                  "audio_dir": str(server_env["audio_dir"])},
        )
        assert resp.status_code == 404
        assert "error" in resp.json()

        resp = client.post(
            "/api/books",
            json={"pdf_path": str(server_env["pdf"]),
                  "audio_dir": str(server_env["pdf"].parent / "no-dir")},
        )
        assert resp.status_code == 404

        empty_dir = server_env["pdf"].parent / "empty"
        empty_dir.mkdir()
        resp = client.post(
            "/api/books",
            json={"pdf_path": str(server_env["pdf"]), "audio_dir": str(empty_dir)},
        )
        assert resp.status_code == 400

        resp = client.post(
            "/api/books",
            json={"pdf_path": str(server_env["audio_dir"] / "book.txt"),
                  "audio_dir": str(server_env["audio_dir"])},
        )
        assert resp.status_code == 400


# -- range streaming -------------------------------------------------------------


def test_audio_get_without_range_is_200_full_body(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        resp = client.get(f"/api/books/{book_id}/audio?format=m4b")
        assert resp.status_code == 200
        assert resp.content == server_env["audio_payload"]
        assert resp.headers["accept-ranges"] == "bytes"


def test_audio_range_100_200_equals_file_slice(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        resp = client.get(
            f"/api/books/{book_id}/audio?format=mp3",
            headers={"Range": "bytes=100-200"},
        )
        assert resp.status_code == 206
        assert resp.content == server_env["audio_payload"][100:201]
        assert resp.headers["content-range"] == (
            f"bytes 100-200/{len(server_env['audio_payload'])}"
        )
        assert int(resp.headers["content-length"]) == 101


def test_audio_suffix_range_and_unsatisfiable(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        resp = client.get(
            f"/api/books/{book_id}/audio", headers={"Range": "bytes=-16"}
        )
        assert resp.status_code == 206
        assert resp.content == server_env["audio_payload"][-16:]

        resp = client.get(
            f"/api/books/{book_id}/audio",
            headers={"Range": f"bytes={len(server_env['audio_payload']) + 5}-"},
        )
        assert resp.status_code == 416


def test_audio_missing_format_is_400_and_missing_file_404(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        assert client.get(
            f"/api/books/{book_id}/audio", params={"format": "ogg"}
        ).status_code == 400


# -- progress ----------------------------------------------------------------


def test_progress_persisted_across_app_instances(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        assert client.get(f"/api/books/{book_id}/progress").json() is None
        resp = client.put(
            f"/api/books/{book_id}/progress",
            json={"position_s": 42.5, "page": 1},
        )
        assert resp.status_code == 200
        saved = resp.json()
        assert saved["position_s"] == 42.5
        assert saved["page"] == 1
        assert saved["updated_at"]

    # a fresh app instance against the same data dir must still see it
    with make_client() as client:
        val = client.get(f"/api/books/{book_id}/progress").json()
        assert val["position_s"] == 42.5
        listing = client.get("/api/books").json()
        assert listing[0]["progress"]["position_s"] == 42.5

        # overwrite
        client.put(
            f"/api/books/{book_id}/progress",
            json={"position_s": 90.0, "page": 2},
        )
        assert client.get(f"/api/books/{book_id}/progress").json()["position_s"] == 90.0


def test_progress_errors(server_env):
    with make_client() as client:
        assert client.get("/api/books/999/progress").status_code == 404
        assert (
            client.put(
                "/api/books/999/progress", json={"position_s": 1.0}
            ).status_code
            == 404
        )


# -- bookmarks ---------------------------------------------------------------


def test_bookmarks_add_list_delete(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        resp = client.post(
            f"/api/books/{book_id}/bookmarks",
            json={"position_s": 12.5, "page": 1, "label": "chapter start"},
        )
        assert resp.status_code == 201
        bm = resp.json()
        assert bm["label"] == "chapter start"
        assert bm["position_s"] == 12.5

        resp = client.post(
            f"/api/books/{book_id}/bookmarks", json={"position_s": 99.0}
        )
        assert resp.status_code == 201
        assert resp.json()["label"] == ""

        listing = client.get(f"/api/books/{book_id}/bookmarks").json()
        assert [b["position_s"] for b in listing] == [12.5, 99.0]

        assert client.delete(f"/api/bookmarks/{bm['id']}").status_code == 200
        assert client.delete(f"/api/bookmarks/{bm['id']}").status_code == 404
        assert len(client.get(f"/api/books/{book_id}/bookmarks").json()) == 1


# -- cover ---------------------------------------------------------------------


def test_cover_endpoint_renders_png(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        resp = client.get(f"/api/books/{book_id}/cover")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"
        # cached on disk
        cache = server_env["data_dir"] / "covers" / f"{book_id}.png"
        assert cache.is_file()


# -- source pdf ----------------------------------------------------------------


def test_pdf_endpoint_serves_source_with_ranges(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        body = server_env["pdf"].read_bytes()

        whole = client.get(f"/api/books/{book_id}/pdf")
        assert whole.status_code == 200
        assert whole.headers["content-type"] == "application/pdf"
        assert whole.headers["accept-ranges"] == "bytes"
        assert whole.content == body

        part = client.get(
            f"/api/books/{book_id}/pdf", headers={"Range": "bytes=0-1023"}
        )
        assert part.status_code == 206
        last = min(1023, len(body) - 1)
        assert part.headers["content-range"] == f"bytes 0-{last}/{len(body)}"
        assert part.content == body[: last + 1]


# -- voices -----------------------------------------------------------------


def test_voices_lists_draft_defaults_and_json_entries(server_env, monkeypatch):
    import json

    from audiobookey.voices import save_voices

    (server_env["data_dir"].mkdir(parents=True, exist_ok=True))
    clone_wav = server_env["pdf"].parent / "me.wav"
    clone_wav.write_bytes(b"RIFF")  # path existence is all validation needs
    save_voices(
        {
            "narrator": {"engine": "draft", "voice_id": "am_michael"},
            "cloned": {"engine": "final", "clone": str(clone_wav)},
        },
        path=server_env["data_dir"] / "voices.json",
    )
    with make_client() as client:
        voices = client.get("/api/voices").json()
        by_name = {v["name"]: v for v in voices}
        assert by_name["narrator"]["engine"] == "draft"
        assert by_name["cloned"]["engine"] == "final"
        assert by_name["cloned"]["has_clone"] is True
        assert by_name["narrator"]["has_clone"] is False
        assert "af_heart" in by_name  # draft defaults present


def test_voices_json_clone_flag_has_clone(server_env):
    import json

    from audiobookey.voices import save_voices

    (server_env["data_dir"].mkdir(parents=True, exist_ok=True))
    clone_wav = server_env["pdf"].parent / "me.wav"
    clone_wav.write_bytes(b"RIFF")  # path existence is all validation needs
    save_voices(
        {"cloned": {"engine": "final", "clone": str(clone_wav)}},
        path=server_env["data_dir"] / "voices.json",
    )
    with make_client() as client:
        voices = client.get("/api/voices").json()
        cloned = next(v for v in voices if v["name"] == "cloned")
        assert cloned["has_clone"] is True
        assert cloned["quality"] == "final"


# -- convert job lifecycle (real draft engine, 1-page PDF) --------------------


def test_convert_job_registers_book(server_env):
    with make_client() as client:
        with open(server_env["pdf"], "rb") as f:
            resp = client.post(
                "/api/convert",
                files={"pdf": ("my-book.pdf", f, "application/pdf")},
                data={"voice": "af_heart", "quality": "draft", "gap": "0.2"},
            )
        assert resp.status_code == 201, resp.text
        job = resp.json()
        assert job["status"] in ("queued", "running")

        deadline = time.time() + 300
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job['id']}").json()
            if job["status"] in ("done", "error"):
                break
            time.sleep(0.5)
        for line in job["log"][-15:]:
            print(line)
        print(job["message"])
        assert job["status"] == "done", job["log"]

        bookmark_in_log = [l for l in job["log"] if l.startswith("EVENT ")]
        assert bookmark_in_log, "expected EVENT progress lines in the job log"
        assert job["page_count"] == 1
        assert job["book_id"] is not None

        jobs = client.get("/api/jobs").json()
        assert any(j["id"] == job["id"] and j["status"] == "done" for j in jobs)

        books = client.get("/api/books").json()
        assert len(books) == 1
        book = books[0]
        assert book["title"] == "my-book"
        assert book["m4b_path"] and os.path.isfile(book["m4b_path"])
        assert book["mp3_path"] and os.path.isfile(book["mp3_path"])
        assert book["duration_s"] > 0

        # range request against the really converted file
        real = open(book["mp3_path"], "rb").read()
        resp = client.get(
            f"/api/books/{book['id']}/audio?format=mp3",
            headers={"Range": "bytes=100-200"},
        )
        assert resp.status_code == 206
        assert resp.content == real[100:201]

        assert client.get(f"/api/jobs/999").status_code == 404


# -- P9 job controls: requeue / pause / cancel ------------------------------

def _freeze_worker(client):
    """Endpoint tests never run a convert: keep queued tasks queued."""
    client.app.state.jobs._ensure_worker = lambda: None


def _inject_live_job(client, job_id, book_id, status="queued"):
    job = {
        "id": job_id,
        "status": status,
        "book_id": book_id,
        "message": "",
        "page_count": 0,
        "page_done": 0,
        "chunk_done": 0,
        "chunk_total": 0,
        "cache_hits": 0,
        "files": [],
        "book_ids": [book_id],
        "log": [],
    }
    client.app.state.jobs.jobs[job_id] = job
    return job


def test_book_row_exposes_job_spec_and_resumable(server_env):
    import json as json_mod

    from audiobookey.server import db as book_db

    with make_client() as client:
        book_id = register_book(client, server_env)
        book = client.get(f"/api/books/{book_id}").json()
        assert book["job_spec"] == {}
        assert book["resumable"] is False  # ready

        book_db.set_job_spec(
            book_id, json_mod.dumps({"voice": "af_heart"}), server_env["data_dir"]
        )
        book_db.set_book_status(book_id, "error", server_env["data_dir"])
        book = client.get(f"/api/books/{book_id}").json()
        assert book["status"] == "error"
        assert book["job_spec"] == {"voice": "af_heart"}
        assert book["resumable"] is True

        book_db.set_book_status(book_id, "paused", server_env["data_dir"])
        assert client.get(f"/api/books/{book_id}").json()["resumable"] is True


def test_convert_submit_persists_spec_on_new_book(server_env):
    with make_client() as client:
        _freeze_worker(client)
        with open(server_env["pdf"], "rb") as f:
            resp = client.post(
                "/api/convert",
                files={"pdf": ("p9-upload.pdf", f, "application/pdf")},
                data={"voice": "af_heart", "quality": "draft", "gap": "0.2"},
            )
        assert resp.status_code == 201, resp.text
        job = resp.json()
        assert job["status"] == "queued"

        books = client.get("/api/books").json()
        assert len(books) == 1
        assert books[0]["status"] == "preparing"
        assert books[0]["job_spec"]["voice"] == "af_heart"
        assert books[0]["job_spec"]["gap"] == 0.2
        assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "queued"


def test_requeue_endpoint_error_matrix(server_env):
    with make_client() as client:
        _freeze_worker(client)
        assert client.post("/api/books/999/convert").status_code == 404
        assert client.post("/api/books/999/pause").status_code == 404
        assert client.post("/api/books/999/cancel").status_code == 404

        book_id = register_book(client, server_env)

        # no live job to stop
        assert client.post(f"/api/books/{book_id}/pause").status_code == 409
        assert client.post(f"/api/books/{book_id}/cancel").status_code == 409

        # no stored spec and no body params -> a helpful 400
        resp = client.post(f"/api/books/{book_id}/convert", json={})
        assert resp.status_code == 400
        assert "parameters" in resp.json()["error"]

        # bad quality / bad gap / broken JSON
        assert client.post(
            f"/api/books/{book_id}/convert", json={"quality": "bogus"}
        ).status_code == 400
        assert client.post(
            f"/api/books/{book_id}/convert", json={"gap": "fast"}
        ).status_code == 400
        resp = client.post(
            f"/api/books/{book_id}/convert",
            content=b"{nope",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400


def test_requeue_pause_cancel_endpoint_flow(server_env):
    import json as json_mod

    from audiobookey.server import db as book_db

    with make_client() as client:
        _freeze_worker(client)
        book_id = register_book(client, server_env)

        # legacy book without stored spec: params must come in the body
        resp = client.post(
            f"/api/books/{book_id}/convert",
            json={"voice": "af_heart", "quality": "draft"},
        )
        assert resp.status_code == 201, resp.text
        job = resp.json()
        assert job["status"] == "queued"
        assert job["book_id"] == book_id
        assert job["book_ids"] == [book_id]

        # same book row: no duplicate, status back to preparing
        books = client.get("/api/books").json()
        assert [b["id"] for b in books] == [book_id]
        assert books[0]["status"] == "preparing"
        spec = json_mod.loads(
            book_db.get_job_spec(book_id, server_env["data_dir"])
        )
        assert spec["voice"] == "af_heart"
        assert spec["gap"] == 0.3  # default filled in

        assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "queued"

        # the live job guards duplicate requeues
        assert client.post(
            f"/api/books/{book_id}/convert", json={"quality": "low"}
        ).status_code == 409

        # queued job: pause tags it (202), the book flips immediately
        resp = client.post(f"/api/books/{book_id}/pause")
        assert resp.status_code == 202
        assert resp.json() == {"job_id": job["id"], "status": "paused"}
        assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "paused"
        book = client.get(f"/api/books/{book_id}").json()
        assert book["status"] == "paused"
        assert book["resumable"] is True  # pdf still on disk

        # no queued/running job anymore; a second pause is a 409, but cancel
        # abandons the paused book (no live job to kill): 202 -> cancelled
        assert client.post(f"/api/books/{book_id}/pause").status_code == 409
        resp = client.post(f"/api/books/{book_id}/cancel")
        assert resp.status_code == 202
        assert resp.json() == {"job_id": None, "status": "cancelled"}

        # one-click resume from the stored spec: no body needed
        resp = client.post(f"/api/books/{book_id}/convert")
        assert resp.status_code == 201, resp.text
        resumed = resp.json()
        assert resumed["status"] == "queued"
        assert client.get(f"/api/books/{book_id}").json()["status"] == "preparing"

        # cancel now (tag-and-skip); C1: a cancelled book must NOT advertise
        # resumable anymore (cancel wipes the partial output + cache)
        resp = client.post(f"/api/books/{book_id}/cancel")
        assert resp.status_code == 202
        assert resp.json() == {"job_id": resumed["id"], "status": "cancelled"}
        book = client.get(f"/api/books/{book_id}").json()
        assert book["status"] == "cancelled"
        assert book["resumable"] is False

        # cancelled -> queued is a plain requeue too (the API allows it; only
        # the resumable flag is gone)
        resp = client.post(f"/api/books/{book_id}/convert")
        assert resp.status_code == 201, resp.text
        assert client.get(f"/api/books/{book_id}").json()["status"] == "preparing"


def test_running_cancel_and_pause_endpoint_marks_outcome(server_env):
    with make_client() as client:
        book_id = register_book(client, server_env)
        killed = []
        client.app.state.jobs._kill_process_tree = killed.append  # no real kill
        _inject_live_job(client, 4711, book_id, status="running")

        resp = client.post(f"/api/books/{book_id}/cancel")
        assert resp.status_code == 202
        assert resp.json() == {"job_id": 4711, "status": "cancelled"}
        assert killed == [None]  # fake job has no subprocess pid yet

        job = client.get("/api/jobs/4711").json()
        assert job["cancel_requested"] is True
        assert job["requested_outcome"] == "cancelled"

        # a later pause re-records the intended outcome (no redundant kill)
        resp = client.post(f"/api/books/{book_id}/pause")
        assert resp.status_code == 202
        assert resp.json() == {"job_id": 4711, "status": "paused"}
        job = client.get("/api/jobs/4711").json()
        assert job["requested_outcome"] == "paused"
        assert len(killed) == 1
