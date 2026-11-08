"""FastAPI application implementing the P3 contract (BRIEF-P3-CONTRACT.md).

All routes under /api/*; when a web/ directory exists next to the repo it is
mounted as the static frontend at /.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .. import voices as voices_mod
from . import covers, db, previews
from .jobs import JobConflict, JobInvalidParams, JobManager, JobNotFound, wipe_book_storage

CHUNK_SIZE = 1 << 16  # 64 KiB


def guess_audio_mime(path: Path) -> str:
    return "audio/mp4" if path.suffix.lower() == ".m4b" else "audio/mpeg"

# Draft-engine defaults offered even when data/voices.json is empty.
DRAFT_DEFAULT_VOICES = (
    "af_heart",
    "af_bella",
    "af_nicole",
    "am_michael",
    "am_fenrir",
    "bf_emma",
    "bm_george",
)


# P6B quality tiers: the server accepts the user-facing low/medium/high plus
# the legacy draft/final, which alias onto the same engine tiers (low -> draft,
# medium/high -> final). The submitted value is passed to job_manager.submit
# verbatim; the CLI resolves the engine alias and, for high, threads the best
# encode bitrate through the final-engine encode call (see cli.py + audiobook.py).
QUALITY_VALUES = ("draft", "final", "low", "medium", "high")


def _require_quality(quality: str) -> None:
    if quality not in QUALITY_VALUES:
        raise HTTPException(
            status_code=400,
            detail="quality must be draft, final, low, medium or high",
        )


def create_app(data_dir: str | Path | None = None, repo_root: Path | None = None) -> FastAPI:
    resolved = db.resolve_data_dir(data_dir)
    repo_root = repo_root or Path(__file__).resolve().parents[3]
    db.init_db(resolved)
    job_manager = JobManager(resolved, repo_root=repo_root)

    app = FastAPI(title="audiobookey", version="0.1.0")
    app.state.data_dir = resolved
    app.state.jobs = job_manager

    # Startup sweep: book rows left 'preparing' by a crash can never resume,
    # so nothing can be running yet — surface them as failed conversions.
    # Their partial pages/ output stays in place and stays playable.
    db.fail_preparing_books(resolved)

    # Startup sweep: adopt every orphan chunk cache into the durable shared
    # root BEFORE any job runs. Job ids restart at 1 with the server, and a
    # job's first act is to wipe convert-tmp/job-<id>/ — without this sweep
    # the next session silently destroys an interrupted session's cache.
    job_manager.adopt_all_orphan_caches()

    def _default_voice(quality: str) -> str:
        """Tier-appropriate fallback when the client omits ``voice``.

        Bare Kokoro ids are draft-only (P2): the old "af_heart" default turned
        every default-voice medium/high job into a guaranteed VoiceError deep
        in the convert subprocess. Final-tier jobs fall back to the first
        configured final-engine voice instead; with none configured we keep
        the old default (the job then fails with that explicit message).
        """
        engine = "draft" if quality in ("low", "draft") else "final"
        if engine == "draft":
            return "af_heart"
        voices = voices_mod.load_voices(voices_mod.voices_path(resolved))
        for name, cfg in voices.items():
            if str(cfg.get("engine")) == "final":
                return name
        return "af_heart"

    def _book_or_404(book_id: int) -> dict:
        book = db.get_book(book_id, resolved)
        if book is None:
            raise HTTPException(status_code=404, detail="book not found")
        return book

    # -- health ------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # -- books -------------------------------------------------------------

    @app.get("/api/books")
    def list_books() -> list[dict]:
        return db.list_books(resolved)

    @app.get("/api/books/{book_id}")
    def get_book(book_id: int) -> dict:
        book = db.get_book(book_id, resolved)
        if book is None:
            raise HTTPException(status_code=404, detail="book not found")
        return book

    @app.post("/api/books", status_code=201)
    def register_book(body: dict) -> dict:
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        pdf_path = Path(str(body.get("pdf_path", "") or ""))
        audio_dir = Path(str(body.get("audio_dir", "") or ""))
        if not pdf_path or pdf_path.suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="pdf_path must end in .pdf")
        if not audio_dir:
            raise HTTPException(status_code=400, detail="audio_dir is required")
        if not pdf_path.is_file():
            raise HTTPException(status_code=404, detail=f"pdf not found: {pdf_path}")
        if not audio_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"audio dir not found: {audio_dir}")

        m4b = _find_audio(audio_dir, ".m4b")
        mp3 = _find_audio(audio_dir, ".mp3")
        if m4b is None and mp3 is None:
            raise HTTPException(
                status_code=400,
                detail=f"no .m4b or .mp3 audio file in {audio_dir}",
            )

        duration_s = _probe_duration(m4b or mp3)
        pages = _pdf_pages(pdf_path)
        book = db.add_book(
            title=pdf_path.stem,
            pdf_path=str(pdf_path),
            m4b_path=str(m4b or ""),
            mp3_path=str(mp3 or ""),
            pages=pages,
            duration_s=duration_s,
            data_dir=resolved,
        )
        return book

    @app.delete("/api/books/{book_id}")
    def delete_book(book_id: int) -> dict:
        book = _book_or_404(book_id)
        # Deleting the row while its conversion still runs orphans the job: it
        # keeps synthesizing into outputs/book-<id>/, _register_result finds no
        # row to update, and the finished audio is unreachable. Refuse instead.
        for job in job_manager.jobs.values():
            if job.get("status") not in ("queued", "running"):
                continue
            ids = job.get("book_ids") or [job.get("book_id")]
            if book_id in ids:
                raise HTTPException(
                    status_code=409,
                    detail="book is still converting; delete it after the conversion finishes",
                )
        db.delete_book(book_id, resolved)
        # C1 full disk reclaim: partial outputs, cover PNG, the book's chunk
        # cache dir and any leftover convert tmp. The uploaded pdf is removed
        # only when it lives inside data/uploads/ (resolve + prefix guard):
        # a path outside the data dir is user data and must never be touched.
        pdf_raw = str((book or {}).get("pdf_path") or "")
        uploads_root = (resolved / "uploads").resolve()
        pdf_inside = False
        pdf_resolved: Path | None = None
        if pdf_raw:
            try:
                pdf_resolved = Path(pdf_raw).resolve()
            except OSError:
                pdf_resolved = None
            pdf_inside = bool(
                pdf_resolved is not None
                and pdf_resolved.is_file()
                and uploads_root in pdf_resolved.parents
            )
        wiped = wipe_book_storage(
            resolved,
            pdf_paths=[Path(pdf_raw)] if pdf_raw else [],
            book_ids=[book_id],
        )
        if pdf_inside and pdf_resolved is not None:
            try:
                pdf_resolved.unlink()
                wiped.append(str(pdf_resolved))
            except OSError:
                pass
        return {"ok": True, "id": book_id, "wiped": wiped}

    # -- preparing/publishing pages ------------------------------------------

    def _pages_index(book_id: int) -> dict:
        """data/outputs/book-<id>/pages/index.json, or an empty stub."""
        path = resolved / "outputs" / f"book-{book_id}" / "pages" / "index.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (ValueError, OSError):
            return {}

    @app.get("/api/books/{book_id}/pages")
    def get_book_pages(book_id: int) -> dict:
        book = _book_or_404(book_id)
        index = _pages_index(book_id)
        entries = [e for e in index.get("pages") or [] if isinstance(e, dict)]
        duration_ready = 0.0
        for entry in entries:
            try:
                duration_ready += float(entry.get("duration_s", 0.0) or 0.0)
            except (TypeError, ValueError, AttributeError):
                pass
        db_pages = int(book.get("pages", 0) or 0)
        index_total = int(index.get("total_pages", 0) or 0)
        return {
            "book_id": book_id,
            "status": index.get("status") or book.get("status", "ready"),
            "total_pages": db_pages if db_pages > 0 else index_total,
            "ready_count": len(entries),
            "duration_ready_s": round(duration_ready, 6),
            "pages": [
                {
                    "page": entry.get("page"),
                    "duration_s": entry.get("duration_s", 0.0),
                    "url": f"/api/books/{book_id}/pages/{entry.get('page')}",
                    # page-local sentence timings (start_s relative to this
                    # page's WAV) so live-mode captions have something to read
                    "sentences": [
                        s
                        for s in (entry.get("sentences") or [])
                        if isinstance(s, dict)
                    ],
                }
                for entry in sorted(entries, key=lambda e: e.get("page", 0))
            ],
        }

    @app.get("/api/books/{book_id}/pages/{page_no}")
    def get_book_page_audio(book_id: int, page_no: int, request: Request):
        _book_or_404(book_id)
        if page_no < 1 or page_no > 100000:
            raise HTTPException(status_code=404, detail="page out of range")
        path = (
            resolved / "outputs" / f"book-{book_id}" / "pages"
            / f"page-{page_no:04d}.wav"
        )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="page not ready")
        return _range_file_response(request, path, "audio/wav")

    # -- audio streaming (HTTP Range) ---------------------------------------

    @app.get("/api/books/{book_id}/audio")
    def stream_audio(book_id: int, request: Request, format: str = "m4b"):
        book = _book_or_404(book_id)
        fmt = (format or "m4b").lower()
        if fmt not in ("m4b", "mp3"):
            raise HTTPException(status_code=400, detail="format must be m4b or mp3")
        audio_path = Path(book["m4b_path" if fmt == "m4b" else "mp3_path"])
        if not audio_path or not audio_path.is_file():
            alt = book["mp3_path" if fmt == "m4b" else "m4b_path"]
            audio_path = Path(alt)
            if not alt or not audio_path.is_file():
                raise HTTPException(
                    status_code=404, detail=f"book {book_id} has no {fmt} audio"
                )

        return _range_file_response(request, audio_path, guess_audio_mime(audio_path))

    # -- cover ---------------------------------------------------------------

    @app.get("/api/books/{book_id}/cover")
    def cover(book_id: int) -> Response:
        book = _book_or_404(book_id)
        pdf_path = Path(book["pdf_path"])
        if not pdf_path or not pdf_path.is_file():
            raise HTTPException(status_code=404, detail="book has no PDF")
        png = covers.cover_png(pdf_path, resolved / "covers" / f"{book_id}.png")
        return Response(content=png.read_bytes(), media_type="image/png")

    # -- source PDF (reader pane) --------------------------------------------

    @app.get("/api/books/{book_id}/pdf")
    def book_pdf(book_id: int, request: Request):
        """The original PDF, Range-capable so the reader loads pages lazily."""
        book = _book_or_404(book_id)
        raw = str(book.get("pdf_path") or "")
        pdf_path = Path(raw)
        if not raw or not pdf_path.is_file():
            raise HTTPException(status_code=404, detail="book has no PDF")
        return _range_file_response(request, pdf_path, "application/pdf")

    # -- progress ------------------------------------------------------------

    @app.get("/api/books/{book_id}/progress")
    def get_progress(book_id: int) -> dict | None:
        _book_or_404(book_id)
        return db.get_progress(book_id, resolved)

    @app.put("/api/books/{book_id}/progress")
    def put_progress(book_id: int, body: dict) -> dict:
        _book_or_404(book_id)
        if not isinstance(body, dict) or "position_s" not in body:
            raise HTTPException(status_code=400, detail="position_s is required")
        try:
            position_s = float(body["position_s"])
            page = int(body.get("page", 0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="position_s/page must be numeric")
        saved = db.set_progress(book_id, position_s, page, resolved)
        return saved

    # -- bookmarks -----------------------------------------------------------

    @app.post("/api/books/{book_id}/bookmarks", status_code=201)
    def add_bookmark(book_id: int, body: dict) -> dict:
        _book_or_404(book_id)
        if not isinstance(body, dict) or "position_s" not in body:
            raise HTTPException(status_code=400, detail="position_s is required")
        try:
            position_s = float(body["position_s"])
            page = int(body.get("page", 0))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="position_s/page must be numeric")
        label = str(body.get("label", "") or "")
        return db.add_bookmark(book_id, position_s, page, label, resolved)

    @app.get("/api/books/{book_id}/bookmarks")
    def list_bookmarks(book_id: int) -> list[dict]:
        book = _book_or_404(book_id)
        return book["bookmarks"]

    @app.delete("/api/bookmarks/{bookmark_id}")
    def delete_bookmark(bookmark_id: int) -> dict:
        if not db.delete_bookmark(bookmark_id, resolved):
            raise HTTPException(status_code=404, detail="bookmark not found")
        return {"ok": True, "id": bookmark_id}

    # -- sentence manifest (P4-B: reader sentence highlighting) ----------------

    @app.get("/api/books/{book_id}/sentences")
    def get_sentences(book_id: int) -> list[dict]:
        book = _book_or_404(book_id)
        m4b_path = book.get("m4b_path") or ""
        manifest_path = Path(m4b_path).parent / "manifest.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="no manifest")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="no manifest")
        sentences = manifest.get("sentences")
        if not isinstance(sentences, list):
            raise HTTPException(status_code=404, detail="no manifest")
        return sentences

    # -- voices ---------------------------------------------------------------

    @app.get("/api/voices")
    def list_voices() -> list[dict]:
        try:
            configured = voices_mod.load_voices(resolved / "voices.json")
        except voices_mod.VoiceError:
            configured = {}
        out: list[dict] = []
        for name, cfg in configured.items():
            engine = cfg.get("engine", "draft")
            out.append(
                {
                    "name": name,
                    "label": name.replace("_", " "),
                    "quality": engine,
                    "engine": engine,
                    "has_clone": bool(cfg.get("clone")),
                }
            )
        known = {v["name"] for v in out}
        for voice_id in DRAFT_DEFAULT_VOICES:
            if voice_id not in known:
                out.append(
                    {
                        "name": voice_id,
                        "label": voice_id.replace("_", " "),
                        "quality": "draft",
                        "engine": "draft",
                        "has_clone": False,
                    }
                )
        return out

    @app.post("/api/voices/{name}/preview")
    def preview_voice(name: str) -> Response:
        """Render previews.SAMPLE_TEXT with the named voice's own engine.

        Voice resolution mirrors the convert path: configured voices in
        data/voices.json first (their ``engine`` decides draft vs final and
        the config is passed to the engine verbatim), then the draft-default
        Kokoro voice ids. Error bodies use the contract's ``{\"detail\": ...}``
        shape (returned directly, bypassing the global {\"error\": ...}
        handler, which stays in place for the existing routes).
        """
        try:
            configured = voices_mod.load_voices(resolved / "voices.json")
        except voices_mod.VoiceError:
            configured = {}
        if name in configured:
            cfg = dict(configured[name])
            engine_kind = str(cfg.get("engine", "draft"))
        elif name in DRAFT_DEFAULT_VOICES:
            cfg = {"engine": "draft", "voice_id": name}
            engine_kind = "draft"
        else:
            return JSONResponse(
                status_code=404, content={"detail": "unknown voice"}
            )

        cache_dir = resolved / "outputs" / "previews"
        wav_path = previews.preview_path(cache_dir, name, engine_kind)
        if wav_path.is_file():
            # Cache hit: no engine is constructed or consulted at all.
            return Response(
                content=wav_path.read_bytes(), media_type="audio/wav"
            )
        try:
            samples, sample_rate = previews.synthesize_preview(engine_kind, cfg)
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"detail": f"preview synthesis failed: {exc}"},
            )
        try:
            data = previews.write_wav_bytes(wav_path, samples, sample_rate)
        except (OSError, ValueError) as exc:
            return JSONResponse(
                status_code=500,
                content={"detail": f"could not encode preview wav: {exc}"},
            )
        return Response(content=data, media_type="audio/wav")

    # -- convert jobs -----------------------------------------------------------

    @app.post("/api/convert", status_code=201)
    async def convert(
        pdf: UploadFile = File(...),
        voice: str = Form(""),
        # Default stays "draft" for backwards compatibility with old clients
        # that omit the field entirely; the web UI always sends an explicit
        # low/medium/high (medium preselected).
        quality: str = Form("draft"),
        gap: float = Form(0.3),
        lexicon_text: str = Form(""),
        ocr: bool = Form(False),
    ) -> dict:
        _require_quality(quality)
        if not voice.strip():
            voice = _default_voice(quality)
        if pdf.filename is None or not pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="upload must be a .pdf")

        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, dir=str(resolved)
        ) as tmp:
            upload_path = Path(tmp.name)
            shutil.copyfileobj(pdf.file, tmp)
        try:
            return job_manager.submit(
                upload_path,
                voice=voice,
                quality=quality,
                gap=gap,
                lexicon_text=lexicon_text or None,
                original_name=pdf.filename,
                ocr=ocr,
            )
        finally:
            # C1: the upload temp (data/tmp*.pdf) is a scratch copy —
            # submit() has already copied the pdf into data/uploads/ on
            # success, and on failure nothing references it. Unlinking in
            # finally covers success, cancel and failure alike.
            upload_path.unlink(missing_ok=True)

    # -- batch convert (P4-A: one batch Job per submission, serialized files) --

    @app.post("/api/convert/batch", status_code=201)
    async def convert_batch(request: Request) -> dict:
        """Convert several PDFs as ONE serialized batch job.

        Accepts either multipart/form-data with repeated ``pdf`` fields (plus
        optional ``voice``/``quality``/``gap``/``ocr`` fields) or a JSON body
        ``{"pdf_paths": [...], "ocr": true}``. The reply is a single batch
        Job whose ``page_count``/``chunk_total`` aggregate across all files
        and whose ``message`` lists the filenames.
        """
        uploads_dir = resolved / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        content_type = (request.headers.get("content-type") or "").lower()
        pdf_paths: list[Path] = []
        created_uploads: list[Path] = []

        voice = ""
        # Same backwards-compat default as the single-file route above.
        quality = "draft"
        gap = 0.3
        ocr = False

        if "application/json" in content_type:
            try:
                body = await request.json()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid JSON body")
            if not isinstance(body, dict):
                raise HTTPException(status_code=400, detail="body must be a JSON object")
            voice = str(body.get("voice", voice)) or voice
            quality = str(body.get("quality", quality)) or quality
            try:
                gap = float(body.get("gap", gap))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="gap must be a number")
            ocr = bool(body.get("ocr", False))
            raw_paths = body.get("pdf_paths") or []
            if not isinstance(raw_paths, list) or not raw_paths:
                raise HTTPException(
                    status_code=400, detail="pdf_paths must be a non-empty list"
                )
            for entry in raw_paths:
                candidate = Path(str(entry))
                if candidate.suffix.lower() != ".pdf":
                    raise HTTPException(
                        status_code=400, detail=f"not a pdf: {candidate}"
                    )
                if not candidate.is_file():
                    raise HTTPException(
                        status_code=404, detail=f"pdf not found: {candidate}"
                    )
                pdf_paths.append(candidate)
        else:
            form = await request.form()
            uploads = [item for item in form.getlist("pdf") if hasattr(item, "read")]
            voice = str(form.get("voice", voice)) or voice
            quality = str(form.get("quality", quality)) or quality
            try:
                gap = float(str(form.get("gap", gap)))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="gap must be a number")
            ocr = str(form.get("ocr", "false")).strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if not uploads:
                raise HTTPException(
                    status_code=400, detail="post one or more 'pdf' fields"
                )
            import uuid

            for item in uploads:
                name = Path(item.filename or "book.pdf").name
                if not name.lower().endswith(".pdf"):
                    raise HTTPException(status_code=400, detail=f"not a pdf: {name}")
                dest = uploads_dir / f"{Path(name).stem}-{uuid.uuid4().hex[:8]}.pdf"
                with dest.open("wb") as out:
                    shutil.copyfileobj(item.file, out)
                created_uploads.append(dest)
                pdf_paths.append(dest)

        _require_quality(quality)
        if not voice.strip():
            voice = _default_voice(quality)

        try:
            return job_manager.submit(
                pdf_paths,
                voice=voice,
                quality=quality,
                gap=gap,
                ocr=ocr,
                display_names=[
                    Path(item.filename or "book.pdf").name
                    for item in uploads
                ] if "application/json" not in content_type else None,
            )
        except Exception:
            # C1: on failure, remove only the upload copies this request
            # created under data/uploads/ — JSON-mode pdf_paths point at
            # caller-owned files that must never be deleted.
            for created in created_uploads:
                created.unlink(missing_ok=True)
            raise

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return list(job_manager.jobs.values())

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int) -> dict:
        job = job_manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    # -- job controls (P9: requeue / pause / cancel per book) ---------------

    @app.post("/api/books/{book_id}/convert", status_code=201)
    async def convert_book(book_id: int, request: Request) -> dict:
        """Requeue (resume) this book's conversion through the FIFO queue.

        Same book row (no duplicate), pdf reused from the book row, status
        back to preparing; optional params ``{voice, quality, gap,
        lexicon_text, ocr, ocr_lang}`` override the stored job spec. Errors:
        404 no book; 400 no body AND no stored spec, bad quality, bad gap;
        409 a job is already queued/running on this book.
        """
        _book_or_404(book_id)
        body: dict = {}
        raw = await request.body()
        if raw and raw.strip():
            try:
                parsed = json.loads(raw)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="invalid JSON body"
                )
            if not isinstance(parsed, dict):
                raise HTTPException(
                    status_code=400, detail="body must be a JSON object"
                )
            body = parsed

        params: dict = {}
        if "voice" in body:
            params["voice"] = str(body["voice"])
        if "quality" in body:
            quality = str(body["quality"] or "")
            _require_quality(quality)
            params["quality"] = quality
        if "gap" in body:
            try:
                params["gap"] = float(body["gap"])
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="gap must be a number")
        if "lexicon_text" in body:
            params["lexicon_text"] = str(body["lexicon_text"] or "")
        if "ocr" in body:
            params["ocr"] = bool(body["ocr"])
        if "ocr_lang" in body:
            params["ocr_lang"] = str(body["ocr_lang"] or "eng")

        try:
            return job_manager.requeue(book_id, params)
        except JobNotFound:
            raise HTTPException(status_code=404, detail="book not found")
        except JobConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except JobInvalidParams as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/books/{book_id}/pause", status_code=202)
    def pause_book(book_id: int) -> dict:
        """Pause this book's live job (kill-and-mark). 404 no book; 409 no
        queued/running job on this book. Resume = POST convert (requeue)."""
        _book_or_404(book_id)
        try:
            return job_manager.pause(book_id)
        except JobConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/books/{book_id}/cancel", status_code=202)
    def cancel_book(book_id: int) -> dict:
        """Cancel this book's live job; acts immediately, no confirmation.
        404 no book; 409 no queued/running job on this book. A paused book
        has no live job: cancel abandons it (paused -> cancelled)."""
        _book_or_404(book_id)
        try:
            return job_manager.cancel(book_id)
        except JobConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    # -- static frontend at / (optional; the P3-B agent owns web/) ------------

    web_dir = repo_root / "web" / "static"
    if web_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        class NoCacheStatic(StaticFiles):
            """Frontend assets must always be revalidated: a browser caching an
            old app.js makes every frontend bug fix invisible until a hard
            refresh (the "I fixed it but it's still broken" loop). index.html
            also carries ?v= cache-busters as a second layer."""

            async def get_response(self, path: str, scope):  # type: ignore[override]
                response = await super().get_response(path, scope)
                response.headers["Cache-Control"] = "no-cache"
                return response

        app.mount("/", NoCacheStatic(directory=str(web_dir), html=True), name="web")

    # Error bodies follow the contract: {"error": "msg"}
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(HTTPException)
    async def http_error(request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code, content={"error": str(exc.detail)}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": str(exc)})

    return app


# -- helpers -------------------------------------------------------------------


def _find_audio(audio_dir: Path, suffix: str) -> Path | None:
    candidates = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() == suffix)
    return candidates[0] if candidates else None


def _probe_duration(path: Path) -> float:
    from ..audio import ffprobe_info

    try:
        info = ffprobe_info(path)
        return float(info["format"]["duration"])
    except (KeyError, ValueError, RuntimeError, OSError):
        return 0.0


def _pdf_pages(pdf_path: Path) -> int:
    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def _parse_range(header: str, file_size: int) -> tuple[int, int] | None:
    """Parse a single-range Range header -> inclusive (start, end) or None."""
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes=") :].split(",")[0].strip()
    if "-" not in spec:
        return None
    start_s, _, end_s = spec.partition("-")
    try:
        if start_s == "":  # suffix range: last N bytes
            length = int(end_s)
            if length <= 0:
                return None
            return (max(0, file_size - length), file_size - 1)
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return None
    if start >= file_size or end < start:
        return None
    return (start, min(end, file_size - 1))


def _range_file_response(request: Request, path: Path, media_type: str):
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if range_header is None:
        return _whole_file_response(path, file_size, media_type)

    rng = _parse_range(range_header, file_size)
    if rng is None:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    start, end = rng

    def chunks():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                piece = f.read(min(CHUNK_SIZE, remaining))
                if not piece:
                    break
                remaining -= len(piece)
                yield piece

    return StreamingResponse(
        chunks(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        },
    )


def _whole_file_response(path: Path, file_size: int, media_type: str) -> StreamingResponse:
    def chunks():
        with open(path, "rb") as f:
            while True:
                piece = f.read(CHUNK_SIZE)
                if not piece:
                    break
                yield piece

    return StreamingResponse(
        chunks(),
        media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )
