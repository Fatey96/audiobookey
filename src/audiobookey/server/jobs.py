"""Convert job management: single-worker queue running the real CLI pipeline.

Jobs run as a subprocess ``uv run audiobookey convert <pdf> --out <tmp> ...
--reporter lines`` from the repo root. The subprocess emits machine-readable
``EVENT {json}`` lines on stdout (LineReporter), which the worker parses into
job progress fields; all output is kept in a rolling log.

P4-A additions: a submitted *list* of pdfs becomes ONE batch job. Batch files
are converted strictly sequentially inside that job (serialized), with per-file
sub-progress in ``message`` and aggregated ``page_count``/``chunk_total``/
``chunk_done`` counters across all files.

P9 additions: convert parameters survive in ``books.job_spec`` so an
interrupted book can be requeued (resume = requeue through the same FIFO
queue), and live jobs can be paused/cancelled (queued: tag-and-skip; running:
tree-kill the subprocess and record the requested outcome instead of error).
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .. import audio as pipeline_audio
from ..cache import book_id as pdf_content_key
from . import db as book_db

MAX_LOG_LINES = 400

# Job outcomes that a pause/cancel request can ask for; anything else falls
# back to "cancelled" so a stray value never invents a state.
_STOP_OUTCOMES = ("paused", "cancelled")

# Book keys are the first 12 hex chars of sha256(pdf bytes): they name the
# per-book cache directories whose orphans the adoption sweeps relocate.
_BOOK_KEY_RE = re.compile(r"^[0-9a-f]{12}$")


def wipe_book_storage(
    data_dir: Path,
    pdf_paths: list[Path] | tuple = (),
    book_ids: list[int] | tuple = (),
    job_tmp_dir: Path | None = None,
    log=None,
) -> list[str]:
    """C1: reclaim one book's partial disk output.

    Deletes the per-book outputs/ dir, the cover PNG, the PDF's chunk-cache
    dir under data/cache/ (key = content hash of the pdf, same derivation the
    pipeline uses via --cache-root), the job's convert-tmp dir and any
    leftover per-book convert-tmp subtrees. Best-effort by contract: a wipe
    failure is skipped and logged (never corrupt the caller's job/row
    bookkeeping). Returns the paths actually removed.
    """
    data = Path(data_dir)
    wiped: list[str] = []
    keys: list[str] = []

    def drop(path: Path) -> None:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists() or path.is_symlink():
                path.unlink()
            else:
                return
        except OSError as exc:
            if log is not None:
                log(f"wipe: could not remove {path}: {exc}")
            return
        wiped.append(str(path))

    for raw_id in book_ids or ():
        try:
            bid = int(raw_id)
        except (TypeError, ValueError):
            continue
        drop(data / "outputs" / f"book-{bid}")
        drop(data / "covers" / f"{bid}.png")
    for pdf in pdf_paths or ():
        try:
            key = pdf_content_key(pdf)
        except OSError:
            if log is not None:
                log(f"wipe: pdf unreadable; its cache dir stays: {pdf}")
            continue
        keys.append(key)
        drop(data / "cache" / key)
    if job_tmp_dir is not None:
        drop(job_tmp_dir)
    if keys:
        tmp_root = data / "convert-tmp"
        if tmp_root.is_dir():
            for chunks_dir in list(tmp_root.rglob("chunks")):
                parts = chunks_dir.relative_to(tmp_root).parts
                key = next(
                    (p for p in reversed(parts[:-1]) if _BOOK_KEY_RE.match(p)),
                    None,
                )
                if key in keys:
                    drop(tmp_root.joinpath(*parts[: parts.index(key) + 1]))
    return wiped


class JobError(Exception):
    pass


class JobNotFound(JobError):
    """No such book/job (API: 404)."""


class JobConflict(JobError):
    """The book already has a live job, or none to stop (API: 409)."""


class JobInvalidParams(JobError):
    """Missing/invalid convert parameters for a requeue (API: 400)."""


class JobManager:
    """Sequential convert-job runner (one job at a time, FIFO queue)."""

    def __init__(self, data_dir: Path, repo_root: Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.repo_root = Path(repo_root) if repo_root else self._default_repo_root()
        self.jobs: dict[int, dict] = {}
        self._queue: queue.Queue[dict] = queue.Queue()
        self._next_id = 1
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._upload_stems: dict[int, list[str]] = {}

    @staticmethod
    def _default_repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    # -- public API --------------------------------------------------------

    def submit(
        self,
        pdf_path: Path | str | list[Path | str],
        voice: str,
        quality: str,
        gap: float,
        lexicon_text: str | None = None,
        original_name: str | None = None,
        ocr: bool = False,
        ocr_lang: str = "eng",
        display_names: list[str] | None = None,
    ) -> dict:
        """Queue a convert job.

        A single pdf keeps the P3 behaviour: one job, one book. A list of
        pdfs becomes ONE batch job: files are converted one at a time inside
        the job, ``message`` carries per-file sub-progress and lists the
        filenames, and the page/chunk counters aggregate across all files.
        ``display_names`` optionally overrides the per-file book titles (used
        by the batch endpoint to keep the uploaded filenames).
        """
        if isinstance(pdf_path, (list, tuple)):
            pdfs = [Path(p) for p in pdf_path]
        else:
            pdfs = [Path(pdf_path)]
        if not pdfs:
            raise ValueError("at least one pdf is required")

        with self._lock:
            job_id = self._next_id
            self._next_id += 1

        if len(pdfs) == 1:
            stems = [self._sanitize_stem(original_name) or pdfs[0].stem]
        elif display_names and len(display_names) == len(pdfs):
            stems = [self._sanitize_stem(name) or p.stem
                     for name, p in zip(display_names, pdfs)]
        else:
            stems = [self._sanitize_stem(p.stem) for p in pdfs]

        uploads = self.data_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        work_pdfs: list[Path] = []
        for index, (pdf, stem) in enumerate(zip(pdfs, stems), start=1):
            suffix = f"-job{job_id}" if len(pdfs) == 1 else f"-job{job_id}-{index}"
            work_pdf = uploads / f"{stem}{suffix}.pdf"
            shutil.copyfile(pdf, work_pdf)
            work_pdfs.append(work_pdf)

        names = [f"{stem}.pdf" for stem in stems]
        is_batch = len(pdfs) > 1

        # P9: remember the convert parameters on every book row so an
        # interrupted/finished book can be requeued (resumed) without the
        # client replaying them or re-uploading the pdf.
        job_spec = json.dumps(
            {
                "voice": str(voice),
                "quality": str(quality),
                "gap": float(gap),
                "lexicon_text": str(lexicon_text or ""),
                "ocr": bool(ocr),
                "ocr_lang": str(ocr_lang or "eng"),
            },
            ensure_ascii=False,
        )

        # One book row per pdf, created BEFORE queueing so the UI can show a
        # preparing card for the whole batch from the first poll.
        book_ids: list[int] = []
        for work_pdf, stem in zip(work_pdfs, stems):
            pages = self._pdf_page_count(work_pdf)
            book = book_db.add_book(
                title=stem,
                pdf_path=str(work_pdf),
                pages=pages,
                duration_s=0.0,
                status="preparing",
                data_dir=self.data_dir,
            )
            book_ids.append(int(book["id"]))
            book_db.set_job_spec(
                int(book["id"]), job_spec, data_dir=self.data_dir
            )

        job = {
            "id": job_id,
            "status": "queued",
            "book_id": book_ids[0] if book_ids else None,
            "message": f"batch: {', '.join(names)}" if is_batch else "",
            "page_count": 0,
            "page_done": 0,
            "chunk_done": 0,
            "chunk_total": 0,
            "cache_hits": 0,
            "files": names,
            "book_ids": book_ids,
            "log": [],
        }
        self.jobs[job_id] = job

        lexicon_file = None
        if lexicon_text:
            lexicon_file = uploads / f"job-{job_id}.lexicon.txt"
            lexicon_file.write_text(lexicon_text, encoding="utf-8")

        self._queue.put(
            {
                "job": job,
                "pdfs": work_pdfs,
                "stems": stems,
                "voice": voice,
                "quality": quality,
                "gap": gap,
                "lexicon_file": lexicon_file,
                "ocr": bool(ocr),
                "ocr_lang": str(ocr_lang),
                "book_ids": book_ids,
            }
        )
        self._ensure_worker()
        return job

    # -- job controls (P9: requeue / pause / cancel) ------------------------

    def requeue(self, book_id: int, params: dict | None = None) -> dict:
        """Requeue (resume) a book's conversion through the FIFO queue.

        Same book row (no duplicate), same uploaded pdf from the book row,
        status back to preparing. ``params`` (voice/quality/gap/lexicon_text/
        ocr/ocr_lang) override the stored job_spec; a stored spec must exist
        or the request must carry parameters. Orphan caches left by the
        interrupted run are adopted first (see adopt_orphan_caches) so the
        requeued convert synthesizes nothing twice.
        """
        book = book_db.get_book(int(book_id), data_dir=self.data_dir)
        if book is None:
            raise JobNotFound(f"book {book_id} not found")
        if self._live_job_for_book(int(book_id)) is not None:
            raise JobConflict(
                f"book {book_id} already has a queued/running job"
            )

        pdf_path = Path(str(book.get("pdf_path") or ""))
        if not pdf_path.is_file():
            raise JobInvalidParams(
                f"book {book_id} no longer has its pdf on disk;"
                " upload the pdf again"
            )

        spec: dict = book.get("job_spec") if isinstance(
            book.get("job_spec"), dict
        ) else {}
        overrides = {
            str(key): value for key, value in (params or {}).items()
            if value is not None
        }
        if not spec and not overrides:
            raise JobInvalidParams(
                "book has no stored convert parameters and the request"
                " carried none; submit voice/quality with the request body"
            )
        try:
            gap = float(overrides.get("gap", spec.get("gap", 0.3)))
        except (TypeError, ValueError):
            raise JobInvalidParams("gap must be a number")
        merged = {
            "voice": str(overrides.get("voice", spec.get("voice", "")) or ""),
            "quality": str(
                overrides.get("quality", spec.get("quality", "draft"))
                or "draft"
            ),
            "gap": gap,
            "lexicon_text": str(
                overrides.get("lexicon_text", spec.get("lexicon_text", ""))
                or ""
            ),
            "ocr": bool(overrides.get("ocr", spec.get("ocr", False))),
            "ocr_lang": str(
                overrides.get("ocr_lang", spec.get("ocr_lang", "eng"))
                or "eng"
            ),
        }

        with self._lock:
            job_id = self._next_id
            self._next_id += 1

        book_db.set_job_spec(
            int(book_id), json.dumps(merged, ensure_ascii=False),
            data_dir=self.data_dir,
        )
        book_db.set_book_status(int(book_id), "preparing", data_dir=self.data_dir)
        adopted = self.adopt_orphan_caches(pdf_path)

        lexicon_file = None
        if merged["lexicon_text"]:
            uploads = self.data_dir / "uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            lexicon_file = uploads / f"job-{job_id}.lexicon.txt"
            lexicon_file.write_text(
                merged["lexicon_text"], encoding="utf-8"
            )

        job = {
            "id": job_id,
            "status": "queued",
            "book_id": int(book_id),
            "message": (
                f"requeued; adopted {adopted} cached chunks"
                if adopted
                else "requeued"
            ),
            "page_count": 0,
            "page_done": 0,
            "chunk_done": 0,
            "chunk_total": 0,
            "cache_hits": 0,
            "files": [pdf_path.name],
            "book_ids": [int(book_id)],
            "log": [],
        }
        self.jobs[job_id] = job

        self._queue.put(
            {
                "job": job,
                "pdfs": [pdf_path],
                "stems": [self._sanitize_stem(book.get("title")) or pdf_path.stem],
                "voice": merged["voice"],
                "quality": merged["quality"],
                "gap": merged["gap"],
                "lexicon_file": lexicon_file,
                "ocr": merged["ocr"],
                "ocr_lang": merged["ocr_lang"],
                "book_ids": [int(book_id)],
            }
        )
        self._ensure_worker()
        return job

    def pause(self, target: int | dict) -> dict:
        """Pause this book's live job; resume re-runs convert against the
        adoptable cache, so nothing synthesized so far is lost."""
        return self._stop_target(target, "paused")

    def cancel(self, target: int | dict) -> dict:
        """Cancel this book's live job (queued: tag-and-skip; running:
        kill-and-mark). Cancel acts immediately; no confirmation.

        A paused book has no live job to stop; cancel simply abandons it:
        the row flips paused -> cancelled and stays requeueable."""
        try:
            return self._stop_target(target, "cancelled")
        except JobConflict:
            if isinstance(target, dict):
                raise
            book = book_db.get_book(int(target), data_dir=self.data_dir)
            if book is not None and book.get("status") == "paused":
                self._set_books_status([int(target)], "cancelled")
                return {"job_id": None, "status": "cancelled"}
            raise

    def _stop_target(self, target: int | dict, outcome: str) -> dict:
        """Kill-and-mark machinery shared by pause and cancel.

        ``target`` is a book id (int) or the job dict itself. A queued job is
        tagged with the outcome and skipped at the top of ``_run_job`` (no
        queue surgery); a running job records the request and tree-kills the
        convert subprocess so the worker frees VRAM and drains the queue.
        Every book in the targeted job shares the outcome (one job = one
        subprocess; partial batches are not separable).
        """
        if outcome not in _STOP_OUTCOMES:
            raise JobError(f"unsupported stop outcome: {outcome}")
        with self._lock:
            job = self._resolve_live_target(target)
            if job is None:
                raise JobConflict(
                    f"no queued/running job for {target!r}"
                )
            book_ids = job.get("book_ids") or [job.get("book_id")]
            if job.get("status") == "queued":
                job["status"] = outcome
                job["message"] = f"{outcome} while queued"
                self._set_books_status(book_ids, outcome)
                return {"job_id": job["id"], "status": outcome}
            request_was_pending = bool(job.get("cancel_requested"))
            job["cancel_requested"] = True
            job["requested_outcome"] = outcome
            pid = job.get("pid")
        if not request_was_pending:
            # Outside the lock: taskkill can take a moment, and calling it for
            # a dying pid twice is harmless.
            self._kill_process_tree(pid)
        return {"job_id": job["id"], "status": outcome}

    def _resolve_live_target(self, target: int | dict) -> dict | None:
        """The live (queued/running) job for this book id or job dict."""
        for job in list(self.jobs.values()):
            if job.get("status") not in ("queued", "running"):
                continue
            if isinstance(target, dict):
                if target.get("id") == job.get("id"):
                    return job
                continue
            ids = job.get("book_ids") or [job.get("book_id")]
            if int(target) in ids:
                return job
        return None

    def _live_job_for_book(self, book_id: int) -> dict | None:
        job = self._resolve_live_target(int(book_id))
        return job

    def _set_books_status(self, book_ids, status: str) -> None:
        """Best-effort row status update; a bad id never breaks the worker."""
        for book_id in book_ids or []:
            try:
                book_db.set_book_status(
                    int(book_id), status, data_dir=self.data_dir
                )
            except (TypeError, ValueError):
                continue

    def _wipe_after_cancel(self, job: dict, task: dict) -> None:
        """C1: a cancelled job reclaims its partial disk output immediately
        (partial pages, chunk cache, convert tmp). Best-effort by contract:
        a wipe failure must never corrupt the job bookkeeping, so errors are
        swallowed and logged to the job log."""
        try:
            wiped = wipe_book_storage(
                self.data_dir,
                pdf_paths=list(task.get("pdfs") or ()),
                book_ids=list(task.get("book_ids") or ()),
                job_tmp_dir=(
                    Path(self.data_dir) / "convert-tmp" / f"job-{job.get('id')}"
                ),
                log=lambda line: self._log(job, line),
            )
            self._log(job, f"cancelled; wiped {len(wiped)} path(s)")
        except Exception as exc:  # defensive: wipe never breaks bookkeeping
            self._log(job, f"cancelled; wipe failed (ignored): {exc}")

    def _outcome_for_stop(self, job: dict) -> str | None:
        """The pause/cancel outcome requested for this job, if any."""
        with self._lock:
            if not job.get("cancel_requested"):
                return None
            outcome = str(job.get("requested_outcome") or "cancelled")
        return outcome if outcome in _STOP_OUTCOMES else "cancelled"

    def _kill_process_tree(self, pid: int | None) -> None:
        """Tree-kill a convert subprocess (ffmpeg grand-children included)."""
        if pid is None or int(pid) <= 0:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                os.kill(int(pid), signal.SIGKILL)
        except OSError:
            # A dying/died pid must never fail a pause/cancel request.
            pass

    def adopt_orphan_caches(self, pdf_path: Path | str) -> int:
        """Move orphan chunk wavs from interrupted converts into the shared
        cache root (``<data>/cache/<book-key>/chunks``), skipping existing
        names. Returns how many files were adopted.

        Interrupted jobs kept their whole chunk cache under
        ``<data>/convert-tmp/job-<id>/cache/<book-key>/chunks``
        (legacy layouts may nest extra levels between cache/ and the book
        key). A cache directory belongs to this book only when the book key
        matches the content hash of the book's pdf, so wavs are never
        adopted across books. Cache keys are content hashes: an identical
        name is an identical chunk.
        """
        key = pdf_content_key(pdf_path)
        dest_dir = self.data_dir / "cache" / key / "chunks"
        dest_dir.mkdir(parents=True, exist_ok=True)
        return self._adopt_caches_for(key).get(key, 0)

    def adopt_all_orphan_caches(self) -> dict[str, int]:
        """Startup sweep: adopt EVERY orphan cache before any job runs.

        Job ids restart from 1 on every server restart, and ``_run_job``
        wipes ``convert-tmp/job-<id>`` before spawning — so the next
        session can destroy an interrupted session's leftover chunk cache
        by pure id collision. Moving all orphans into the durable shared
        root at startup makes that wipe harmless. Returns
        ``{book-key: adopted wav count}`` for keys with at least one move.
        """
        return self._adopt_caches_for(None)

    def _adopt_caches_for(self, only_key: str | None) -> dict[str, int]:
        """Shared walker behind the targeted and startup adoption sweeps.

        ``only_key`` restricts adoption to that book key; ``None`` adopts
        every hex book key found under convert-tmp/.
        """
        tmp_root = self.data_dir / "convert-tmp"
        moved: dict[str, int] = {}
        if not tmp_root.is_dir():
            return moved
        for chunks_dir in tmp_root.rglob("chunks"):
            if not chunks_dir.is_dir():
                continue
            # The book key names an ancestor directory of chunks/.
            parts = chunks_dir.relative_to(tmp_root).parts
            key = next(
                (p for p in reversed(parts[:-1]) if _BOOK_KEY_RE.match(p)),
                None,
            )
            if key is None or (only_key is not None and key != only_key):
                continue
            dest_dir = self.data_dir / "cache" / key / "chunks"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for wav in chunks_dir.glob("*.wav"):
                dest = dest_dir / wav.name
                if dest.exists():
                    continue
                shutil.move(str(wav), dest)
                moved[key] = moved.get(key, 0) + 1
        return moved

    # -- worker ------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_worker, name="convert-jobs", daemon=True
        )
        self._thread.start()

    def _run_worker(self) -> None:
        while True:
            task = self._queue.get()
            try:
                self._run_job(task)
            except Exception as exc:  # keep the worker alive whatever happens
                job = task["job"]
                outcome = self._outcome_for_stop(job)
                if outcome is None:
                    outcome = "error"
                    message = f"{type(exc).__name__}: {exc}"
                else:
                    message = f"{outcome} by user"
                job["status"] = outcome
                job["message"] = message
                # Record the outcome on the book rows in-session instead of
                # leaving them 'preparing' for the next restart sweep.
                self._set_books_status(task.get("book_ids"), outcome)
                if outcome == "cancelled":
                    self._wipe_after_cancel(job, task)

    def _log(self, job: dict, line: str) -> None:
        job["log"].append(line)
        if len(job["log"]) > MAX_LOG_LINES:
            del job["log"][: len(job["log"]) - MAX_LOG_LINES]

    def _file_message(
        self, file_prefix: str, index: int, total: int, name: str, detail: str
    ) -> str:
        sub = f"file {index}/{total} ({name}): {detail}" if total > 1 else detail
        if file_prefix and total > 1:
            return f"{file_prefix} | {sub}"
        return sub

    def _run_job(self, task: dict) -> None:
        job: dict = task["job"]
        # P9 tag-and-skip: a queued job paused/cancelled before the worker
        # reached it never starts a subprocess; the queue keeps draining
        # without any surgery. The check-and-claim below shares the lock with
        # _stop_target's queued tag so the two cannot interleave.
        skip: str | None
        with self._lock:
            pending_status = job.get("status")
            if pending_status in _STOP_OUTCOMES:
                skip = pending_status
            else:
                skip = None
                job["status"] = "running"
        if skip is not None:
            self._log(job, f"skipped: job {skip} while queued")
            self._set_books_status(task.get("book_ids"), skip)
            if skip == "cancelled":
                self._wipe_after_cancel(job, task)
            return
        out_dir = Path(self.data_dir) / "convert-tmp" / f"job-{job['id']}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        pdfs: list[Path] = task["pdfs"]
        stems: list[str] = task["stems"]
        names: list[str] = [p.name for p in pdfs]
        self._upload_stems[job["id"]] = stems
        file_prefix = f"batch: {', '.join(names)}" if len(pdfs) > 1 else ""

        # Rows were pre-created at submit; the worker only fills them in.
        book_ids: list[int] = task.get("book_ids") or []

        for index, work_pdf in enumerate(pdfs, start=1):
            name = names[index - 1]
            job["message"] = self._file_message(
                file_prefix, index, len(pdfs), name, "starting convert"
            )

            cmd = [
                sys.executable,
                "-m",
                "audiobookey",
                "convert",
                str(work_pdf),
                "--out",
                str(out_dir),
                # P9: park the chunk cache under the shared data dir so a
                # requeued (resumed) convert hits every chunk earlier runs
                # synthesized instead of the per-job tmp dir.
                "--cache-root",
                str(self.data_dir / "cache"),
                "--voice",
                str(task["voice"]),
                "--quality",
                str(task["quality"]),
                "--gap",
                str(task["gap"]),
                "--reporter",
                "lines",
            ]
            if task.get("lexicon_file") is not None:
                cmd += ["--lexicon", str(task["lexicon_file"])]
            if task.get("ocr"):
                cmd += ["--ocr"]
                if task.get("ocr_lang"):
                    cmd += ["--ocr-lang", str(task["ocr_lang"])]

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError as exc:
                outcome = self._outcome_for_stop(job)
                if outcome is not None:
                    job["status"] = outcome
                    job["message"] = (
                        f"{outcome} by user; convert failed to start"
                    )
                else:
                    job["status"] = "error"
                    job["message"] = f"failed to start convert: {exc}"
                # P9 bug fix: record the outcome on this book's row in-session,
                # not on the next restart sweep.
                self._set_books_status(
                    book_ids[index - 1:], job["status"]
                )
                return

            # P9: remember the pid so pause/cancel can tree-kill the convert
            # while it runs, and honor a request that landed between the
            # dequeue and this spawn by killing the fresh process here.
            # getattr: test doubles for Popen may not carry a pid.
            job["pid"] = getattr(proc, "pid", None)
            if job.get("cancel_requested"):
                self._kill_process_tree(job["pid"])

            assert proc.stdout is not None
            done_before = job["chunk_done"]
            book_id = book_ids[index - 1] if index - 1 < len(book_ids) else None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                self._log(job, line)
                if line.startswith("EVENT "):
                    try:
                        payload = json.loads(line[len("EVENT ") :])
                    except (json.JSONDecodeError, ValueError):
                        continue

                    self._apply_event(
                        job,
                        payload,
                        file_prefix,
                        index,
                        len(pdfs),
                        name,
                        done_before,
                        book_id,
                    )
            proc.wait()
            self._log(job, f"[exit code {proc.returncode}]")

            if proc.returncode != 0:
                outcome = self._outcome_for_stop(job)
                if outcome is not None:
                    # Pause/cancel was requested and the subprocess died for
                    # that reason: record the intended outcome, never error.
                    job["status"] = outcome
                    job["message"] = (
                        f"{outcome} by user; convert exited with code"
                        f" {proc.returncode}"
                    )
                else:
                    job["status"] = "error"
                    job["message"] = (
                        f"convert exited with code {proc.returncode};"
                        " see job log for details"
                    )
                    if len(pdfs) > 1:
                        job["message"] = (
                            f"file {index}/{len(pdfs)} ({name}) failed; "
                            + job["message"]
                            + "; earlier files in this batch were registered"
                            if index > 1
                            else job["message"]
                        )
                # P9 bug fix: a convert failure must flip this book to its
                # outcome in-session; it used to stay 'preparing' until the
                # restart sweep. Books registered earlier in the batch keep
                # their ready rows.
                self._set_books_status(
                    book_ids[index - 1:], job["status"]
                )
                if outcome == "cancelled":
                    self._wipe_after_cancel(job, task)
                return

            try:
                self._register_result(
                    job, work_pdf, out_dir, stems[index - 1], book_id
                )
            except (JobError, OSError, ValueError) as exc:
                job["status"] = "error"
                detail = f"convert finished but registration failed: {exc}"
                job["message"] = (
                    self._file_message(
                        file_prefix, index, len(pdfs), name, detail
                    )
                    if len(pdfs) > 1
                    else detail
                )
                if book_id is not None:
                    book_db.set_book_status(
                        int(book_id), "error", data_dir=self.data_dir
                    )
                return

            job["book_id"] = book_id  # P3 compat: last registered book
            job["book_ids"] = list(book_ids)
            job["message"] = self._file_message(
                file_prefix,
                index,
                len(pdfs),
                name,
                f"done; registered book {book_id}",
            )

        job["book_ids"] = book_ids
        job["status"] = "done"
        job["message"] = (
            (
                f"batch complete: {', '.join(names)}"
                f" ({len(book_ids)}/{len(pdfs)} books)"
            )
            if len(pdfs) > 1
            else f"done; registered book {job['book_id']}"
        )
        shutil.rmtree(out_dir, ignore_errors=True)

    def _apply_event(
        self,
        job: dict,
        payload: dict,
        file_prefix: str,
        index: int,
        total: int,
        name: str,
        done_before: int,
        book_id: int | None,
    ) -> None:
        """Fold one reporter EVENT into the job dict (unit-testable)."""
        if "page_ready" in payload:
            self._publish_page(
                job, payload["page_ready"], file_prefix, index, total, name, book_id
            )
        elif "phase" in payload:
            job["message"] = self._file_message(
                file_prefix,
                index,
                total,
                name,
                str(payload["phase"]),
            )
        elif "start_chunks" in payload:
            start = payload["start_chunks"]
            job["page_count"] += int(start.get("page_count", 0))
            job["chunk_total"] += int(start.get("chunk_total", 0))
        elif "chunk_done" in payload:
            job["chunk_done"] = max(
                job["chunk_done"],
                done_before + int(payload["chunk_done"]),
            )
            job["message"] = self._file_message(
                file_prefix,
                index,
                total,
                name,
                f"synthesizing chunk {payload['chunk_done']}",
            )
        elif "update" in payload:
            update = payload["update"]
            job["cache_hits"] = int(update.get("cache_hits", 0))
        elif "warning" in payload:
            self._log(job, f"warning: {payload['warning']}")

    def _publish_page(
        self,
        job: dict,
        page_ready: dict,
        file_prefix: str,
        index: int,
        total: int,
        name: str,
        book_id: int | None,
    ) -> None:
        if book_id is None:
            return
        page = int(page_ready.get("page", index) or index)
        pages_dir = self.data_dir / "outputs" / f"book-{book_id}" / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        wav_src = Path(str(page_ready.get("wav", "")))
        wav_dest = pages_dir / f"page-{page:04d}.wav"
        if wav_src.is_file():
            shutil.copyfile(wav_src, wav_dest)

        index_path = pages_dir / "index.json"
        entries: list[dict] = []
        total_pages = int(page_ready.get("total_pages", 0) or 0)
        if index_path.is_file():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    entries = [
                        e for e in (data.get("pages") or []) if isinstance(e, dict)
                    ]
                    total_pages = int(data.get("total_pages", 0) or total_pages)
            except (json.JSONDecodeError, OSError):
                entries = []

        entry = {
            "page": page,
            "file": wav_dest.name,
            "duration_s": float(page_ready.get("duration_s", 0.0) or 0.0),
            "sentences": list(page_ready.get("sentences") or []),
        }
        entries = [e for e in entries if int(e.get("page", -1)) != page]
        entries.append(entry)
        entries.sort(key=lambda e: int(e.get("page", 0)))

        tmp = index_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "book_id": book_id,
                    "status": "preparing",
                    "total_pages": total_pages,
                    "pages": entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, index_path)

        job["page_done"] += 1
        job["message"] = self._file_message(
            file_prefix,
            index,
            total,
            name,
            f"page {page} of {total_pages} ready",
        )

    def _register_result(
        self, job: dict, work_pdf: Path, out_dir: Path, stem: str, book_id: int
    ) -> int:
        """Update the pre-created book row (same id) and publish its outputs."""
        m4b = out_dir / f"{work_pdf.stem}.m4b"
        mp3 = out_dir / f"{work_pdf.stem}.mp3"
        if not m4b.is_file():
            raise JobError(f"no .m4b produced for {work_pdf.name}")
        mp3 = mp3 if mp3.is_file() else None

        pages = 0
        import pymupdf

        doc = pymupdf.open(work_pdf)
        try:
            pages = doc.page_count
        finally:
            doc.close()

        duration_s = 0.0
        try:
            info = pipeline_audio.ffprobe_info(m4b)
            duration_s = float(info["format"]["duration"])
        except (KeyError, ValueError, RuntimeError, OSError):
            pass

        final_dir = self.data_dir / "outputs" / f"book-{book_id}"
        final_dir.mkdir(parents=True, exist_ok=True)

        def _publish(src: Path, dest: Path) -> None:
            """Copy beside the target, then atomically replace it.

            Never wipe the book directory here. Progressive playback means the
            web player may be streaming a published page WAV out of this very
            directory; on Windows an open handle makes shutil.rmtree raise
            WinError 32, which used to fail the whole registration after a
            perfectly good conversion (book stuck in status=error, no m4b).
            """
            tmp = dest.with_name(dest.name + ".new")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dest)

        final_m4b = final_dir / m4b.name
        _publish(m4b, final_m4b)
        final_mp3 = None
        if mp3 is not None:
            final_mp3 = final_dir / mp3.name
            _publish(mp3, final_mp3)
        # Sentence manifest for the reader UI (P4): written next to the m4b
        # by convert(); books converted before P4 simply lack it.
        manifest_src = m4b.parent / "manifest.json"
        if manifest_src.is_file():
            _publish(manifest_src, final_dir / "manifest.json")

        # Row update last: paths must point at files that already exist, and
        # status flips to ready only after the copies succeeded.
        book_db.update_book_paths(
            book_id,
            pdf_path=str(self._original_pdf_path(work_pdf)),
            m4b_path=str(final_m4b),
            mp3_path=str(final_mp3 or ""),
            pages=pages,
            duration_s=duration_s,
            data_dir=self.data_dir,
        )
        book_db.set_book_status(book_id, "ready", data_dir=self.data_dir)

        # m4b supersedes the published per-page WAVs; once the finalize copy
        # succeeded the pages/ dir can go (stats fall back to the db row).
        pages_dir = final_dir / "pages"
        index_path = pages_dir / "index.json"
        if index_path.is_file():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data["status"] = "ready"
                    tmp = index_path.with_suffix(".json.tmp")
                    tmp.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    os.replace(tmp, index_path)
            except (json.JSONDecodeError, OSError):
                pass
        # Best-effort page cleanup. A listener may still hold one of these WAVs
        # open mid-stream (Windows: WinError 32), so retry briefly and then
        # leave whatever is left for the next publish/sweep. Cleanup must never
        # fail a registration that already succeeded.
        for _ in range(8):
            shutil.rmtree(pages_dir, ignore_errors=True)
            if not pages_dir.exists():
                break
            time.sleep(0.25)
        return book_id

    @staticmethod
    def _sanitize_stem(name: str | None) -> str:
        if not name:
            return ""
        stem = Path(name).stem
        return re.sub(r"[^A-Za-z0-9_-]+", "-", stem)[:80]

    @staticmethod
    def _pdf_page_count(pdf: Path) -> int:
        """Page count for the pre-created book row; 0 on any failure."""
        try:
            import pymupdf

            doc = pymupdf.open(pdf)
            try:
                return int(doc.page_count)
            finally:
                doc.close()
        except Exception:
            return 0

    def _original_pdf_path(self, work_pdf: Path) -> Path:
        """The upload copy in data/uploads serves as the book's PDF."""
        return work_pdf
