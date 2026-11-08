"""SQLite persistence for the web server (books, progress, bookmarks)."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    pdf_path    TEXT NOT NULL DEFAULT '',
    m4b_path    TEXT NOT NULL DEFAULT '',
    mp3_path    TEXT NOT NULL DEFAULT '',
    pages       INTEGER NOT NULL DEFAULT 0,
    duration_s  REAL NOT NULL DEFAULT 0.0,
    created_at  TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE IF NOT EXISTS progress (
    book_id    INTEGER PRIMARY KEY REFERENCES books(id) ON DELETE CASCADE,
    position_s REAL NOT NULL DEFAULT 0.0,
    page       INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE TABLE IF NOT EXISTS bookmarks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id    INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    position_s REAL NOT NULL,
    page       INTEGER NOT NULL DEFAULT 0,
    label      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the server data dir (explicit arg, then env, then data/).

    The explicit argument must win: every db call re-resolves, so an ambient
    AUDIOBOOKEY_DATA_DIR overriding a caller-pinned data_dir would fork the
    writer (JobManager) from the reader (list_books) into different databases.
    """
    if data_dir:
        return Path(data_dir)
    env = os.environ.get("AUDIOBOOKEY_DATA_DIR")
    if env:
        return Path(env)
    from ..voices import find_data_dir

    return find_data_dir()


def db_path(data_dir: str | Path | None = None) -> Path:
    return resolve_data_dir(data_dir) / "audiobookey.db"


def connect(data_dir: str | Path | None = None) -> sqlite3.Connection:
    base = resolve_data_dir(data_dir)
    base.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(base / "audiobookey.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(data_dir: str | Path | None = None) -> Path:
    conn = connect(data_dir)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        # Idempotent migration for databases created before the status
        # column existed. Safe to call on every boot and twice in a row:
        # the column is only added when PRAGMA reports it missing.
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(books)")
        }
        if "status" not in columns:
            conn.execute(
                "ALTER TABLE books ADD COLUMN status TEXT NOT NULL"
                " DEFAULT 'ready'"
            )
            conn.commit()
        # P9 job controls: the convert parameters used for a book survive
        # restarts and power requeue (resume) without re-uploading the pdf.
        if "job_spec" not in columns:
            conn.execute(
                "ALTER TABLE books ADD COLUMN job_spec TEXT NOT NULL"
                " DEFAULT ''"
            )
            conn.commit()
    finally:
        conn.close()
    return db_path(data_dir)


# Statuses a book can be resumed (requeued) from; 'ready' books re-convert
# normally, live jobs are guarded by the API layer. C1: cancelled books are
# terminal — a cancel wipes the partial output and cache, so there is nothing
# to resume and the row must not advertise resumable.
_RESUMABLE_STATUSES = ("error", "paused")


def _pdf_exists(raw: str, data_dir: str | Path | None) -> bool:
    if not raw:
        return False
    path = Path(raw)
    if path.is_absolute():
        return path.is_file()
    if data_dir and (resolve_data_dir(data_dir) / path).is_file():
        return True
    return path.is_file()


def _row_to_book(
    row: sqlite3.Row,
    progress: Any,
    bookmarks: list[dict],
    data_dir: str | Path | None = None,
) -> dict:
    status = row["status"] if "status" in row.keys() else "ready"
    raw_spec = row["job_spec"] if "job_spec" in row.keys() else ""
    try:
        spec = json.loads(raw_spec) if raw_spec else {}
    except ValueError:
        spec = {}
    book = {
        "id": row["id"],
        "title": row["title"],
        "pdf_path": row["pdf_path"],
        "m4b_path": row["m4b_path"],
        "mp3_path": row["mp3_path"],
        "pages": row["pages"],
        "cover_path": f"/api/books/{row['id']}/cover",
        "duration_s": row["duration_s"],
        "status": status,
        # P9: the stored convert parameters (parsed; {} when absent) plus the
        # one-click resume flag the UI keys its Resume/Requeue actions on.
        "job_spec": spec if isinstance(spec, dict) else {},
        "resumable": status in _RESUMABLE_STATUSES
        and _pdf_exists(row["pdf_path"], data_dir),
        "progress": progress,
        "bookmarks": bookmarks,
    }
    return book


def _get_progress(conn: sqlite3.Connection, book_id: int) -> dict | None:
    row = conn.execute(
        "SELECT book_id, position_s, page, updated_at FROM progress WHERE book_id = ?",
        (book_id,),
    ).fetchone()
    if row is None:
        return None
    return {
        "position_s": row["position_s"],
        "page": row["page"],
        "updated_at": row["updated_at"],
    }


def _get_bookmarks(conn: sqlite3.Connection, book_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, position_s, page, label, created_at FROM bookmarks "
        "WHERE book_id = ? ORDER BY position_s",
        (book_id,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "position_s": r["position_s"],
            "page": r["page"],
            "label": r["label"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def _ready_pages_index(book_id: int, data_dir: str | Path | None) -> dict | None:
    """Published page entries from data/outputs/book-<id>/pages/index.json."""
    base = resolve_data_dir(data_dir)
    path = base / "outputs" / f"book-{book_id}" / "pages" / "index.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        return None
    return data


def _attach_ready_stats(book: dict, data_dir: str | Path | None) -> dict:
    index = _ready_pages_index(book["id"], data_dir)
    if index is None:
        ready_pages = 0
        duration_ready_s = 0.0
        if book.get("status") == "ready":
            ready_pages = book.get("pages", 0) or 0
            duration_ready_s = float(book.get("duration_s", 0.0) or 0.0)
    else:
        ready_pages = 0
        duration_ready_s = 0.0
        for entry in index["pages"]:
            try:
                ready_pages += 1
                duration_ready_s += float(entry.get("duration_s", 0.0) or 0.0)
            except (TypeError, ValueError, AttributeError):
                pass
    book["ready_pages"] = int(ready_pages)
    book["duration_ready_s"] = float(duration_ready_s)
    return book


def list_books(data_dir: str | Path | None = None) -> list[dict]:
    conn = connect(data_dir)
    try:
        books: list[dict] = []
        for row in conn.execute("SELECT * FROM books ORDER BY id"):
            book = _row_to_book(
                row,
                _get_progress(conn, row["id"]),
                _get_bookmarks(conn, row["id"]),
                data_dir,
            )
            books.append(_attach_ready_stats(book, data_dir))
        return books
    finally:
        conn.close()


def get_book(book_id: int, data_dir: str | Path | None = None) -> dict | None:
    conn = connect(data_dir)
    try:
        row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
        if row is None:
            return None
        book = _row_to_book(
            row,
            _get_progress(conn, book_id),
            _get_bookmarks(conn, book_id),
            data_dir,
        )
    finally:
        conn.close()
    return _attach_ready_stats(book, data_dir)


def add_book(
    title: str,
    pdf_path: str = "",
    m4b_path: str = "",
    mp3_path: str = "",
    pages: int = 0,
    duration_s: float = 0.0,
    status: str = "ready",
    data_dir: str | Path | None = None,
) -> dict:
    conn = connect(data_dir)
    try:
        cur = conn.execute(
            "INSERT INTO books (title, pdf_path, m4b_path, mp3_path, pages,"
            " duration_s, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (title, pdf_path, m4b_path, mp3_path, pages, duration_s, status),
        )
        conn.commit()
        return get_book(cur.lastrowid, data_dir)
    finally:
        conn.close()


def update_book_paths(
    book_id: int,
    pdf_path: str,
    m4b_path: str,
    mp3_path: str,
    pages: int,
    duration_s: float,
    data_dir: str | Path | None = None,
) -> None:
    conn = connect(data_dir)
    try:
        conn.execute(
            "UPDATE books SET pdf_path = ?, m4b_path = ?, mp3_path = ?, "
            "pages = ?, duration_s = ?, status = 'ready' WHERE id = ?",
            (pdf_path, m4b_path, mp3_path, pages, duration_s, book_id),
        )
        conn.commit()
    finally:
        conn.close()


def set_book_status(
    book_id: int,
    status: str,
    data_dir: str | Path | None = None,
) -> None:
    conn = connect(data_dir)
    try:
        conn.execute("UPDATE books SET status = ? WHERE id = ?", (status, book_id))
        conn.commit()
    finally:
        conn.close()


def set_job_spec(
    book_id: int,
    spec_json: str,
    data_dir: str | Path | None = None,
) -> None:
    """Persist the convert parameters for this book's next requeue (P9)."""
    conn = connect(data_dir)
    try:
        conn.execute(
            "UPDATE books SET job_spec = ? WHERE id = ?", (spec_json, book_id)
        )
        conn.commit()
    finally:
        conn.close()


def get_job_spec(book_id: int, data_dir: str | Path | None = None) -> str:
    """Stored convert parameters as raw JSON ('' when never submitted)."""
    conn = connect(data_dir)
    try:
        row = conn.execute(
            "SELECT job_spec FROM books WHERE id = ?", (book_id,)
        ).fetchone()
        if row is None:
            return ""
        return str(row["job_spec"] or "")
    finally:
        conn.close()


def fail_preparing_books(data_dir: str | Path | None = None) -> None:
    conn = connect(data_dir)
    try:
        conn.execute(
            "UPDATE books SET status = 'error' WHERE status = 'preparing'"
        )
        conn.commit()
    finally:
        conn.close()


def delete_book(book_id: int, data_dir: str | Path | None = None) -> bool:
    conn = connect(data_dir)
    try:
        cur = conn.execute("DELETE FROM books WHERE id = ?", (book_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def set_progress(
    book_id: int,
    position_s: float,
    page: int,
    data_dir: str | Path | None = None,
) -> dict | None:
    conn = connect(data_dir)
    try:
        conn.execute(
            "INSERT INTO progress (book_id, position_s, page, updated_at)"
            " VALUES (?, ?, ?, STRFTIME('%Y-%m-%dT%H:%M:%fZ', 'now'))"
            " ON CONFLICT(book_id) DO UPDATE SET"
            " position_s = excluded.position_s,"
            " page = excluded.page,"
            " updated_at = excluded.updated_at",
            (book_id, position_s, page),
        )
        conn.commit()
        return _get_progress(conn, book_id)
    finally:
        conn.close()


def get_progress(book_id: int, data_dir: str | Path | None = None) -> dict | None:
    conn = connect(data_dir)
    try:
        return _get_progress(conn, book_id)
    finally:
        conn.close()


def get_bookmark(
    bookmark_id: int, data_dir: str | Path | None = None
) -> dict | None:
    conn = connect(data_dir)
    try:
        row = conn.execute(
            "SELECT id, book_id, position_s, page, label, created_at"
            " FROM bookmarks WHERE id = ?",
            (bookmark_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "book_id": row["book_id"],
            "position_s": row["position_s"],
            "page": row["page"],
            "label": row["label"],
            "created_at": row["created_at"],
        }
    finally:
        conn.close()


def add_bookmark(
    book_id: int,
    position_s: float,
    page: int,
    label: str,
    data_dir: str | Path | None = None,
) -> dict:
    conn = connect(data_dir)
    try:
        cur = conn.execute(
            "INSERT INTO bookmarks (book_id, position_s, page, label)"
            " VALUES (?, ?, ?, ?)",
            (book_id, position_s, page, label),
        )
        conn.commit()
        return get_bookmark(cur.lastrowid, data_dir)
    finally:
        conn.close()


def delete_bookmark(bookmark_id: int, data_dir: str | Path | None = None) -> bool:
    conn = connect(data_dir)
    try:
        cur = conn.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
