"""Progress reporting protocol plus a rich-based console reporter."""

from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class Reporter:
    """Sink for pipeline progress; the pipeline never prints directly."""

    def phase(self, message: str) -> None:  # pragma: no cover - trivial
        pass

    def warning(self, message: str) -> None:  # pragma: no cover - trivial
        pass

    def start_chunks(self, total: int, total_pages: int) -> None:  # pragma: no cover
        pass

    def chunk_done(self) -> None:  # pragma: no cover - trivial
        pass

    def update(self, cache_hits: int, done: int, total: int) -> None:  # pragma: no cover
        pass

    def page_ready(self, page: int, index: int, total_pages: int, wav, duration_s: float, sentences: list) -> None:
        pass

    def __enter__(self) -> "Reporter":
        return self

    def __exit__(self, *exc) -> None:
        pass


class QuietReporter(Reporter):
    pass


class LineReporter(Reporter):
    """Machine-readable reporter: one ``EVENT {json}`` line per callback.

    Used by the web server's convert jobs (subprocess stdout parsing).
    """

    _SENTINEL = object()

    def __init__(self, stream=None) -> None:
        import sys

        self._stream = stream if stream is not None else sys.stdout

    def _emit(self, payload: dict) -> None:
        import json

        self._stream.write("EVENT " + json.dumps(payload, ensure_ascii=False) + "\n")
        self._stream.flush()

    def phase(self, message: str) -> None:
        self._emit({"phase": message})

    def warning(self, message: str) -> None:
        self._emit({"warning": message})

    def start_chunks(self, total: int, total_pages: int) -> None:
        self._emit({"start_chunks": {"chunk_total": total, "page_count": total_pages}})
        self._done = 0

    def chunk_done(self) -> None:
        self._done = getattr(self, "_done", 0) + 1
        self._emit({"chunk_done": self._done})

    def update(self, cache_hits: int, done: int, total: int) -> None:
        self._emit({"update": {"cache_hits": cache_hits, "done": done, "total": total}})

    def page_ready(self, page: int, index: int, total_pages: int, wav, duration_s: float, sentences: list) -> None:
        self._emit(
            {
                "page_ready": {
                    "page": int(page),
                    "index": int(index),
                    "total_pages": int(total_pages),
                    "wav": str(wav),
                    "duration_s": float(duration_s),
                    "sentences": list(sentences),
                }
            }
        )


class RichReporter(Reporter):
    """Shows page progress, cache hit rate and ETA."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._progress: Progress | None = None
        self._task_id = None

    def __enter__(self) -> "RichReporter":
        self._progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("·"),
            TimeElapsedColumn(),
            TextColumn("elapsed"),
            TextColumn("·"),
            TimeRemainingColumn(),
            TextColumn("remaining"),
            console=self.console,
        )
        self._progress.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        assert self._progress is not None
        self._progress.__exit__(*exc)
        self._progress = None

    def phase(self, message: str) -> None:
        if self._progress is not None:
            self._progress.console.print(f"[bold]{message}[/bold]")
        else:  # pragma: no cover
            self.console.print(message)

    def warning(self, message: str) -> None:
        self.console.print(f"[yellow]warning:[/yellow] {message}")

    def start_chunks(self, total: int, total_pages: int) -> None:
        assert self._progress is not None
        self._task_id = self._progress.add_task(
            f"0/{total_pages} pages, 0% cache hits", total=max(total, 1)
        )
        self._total_pages = total_pages

    def chunk_done(self) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id)

    def update(self, cache_hits: int, done: int, total: int) -> None:
        if self._progress is not None and self._task_id is not None:
            rate = 100.0 * cache_hits / total if total else 0.0
            self._progress.update(
                self._task_id,
                description=f"{done} chunks, {rate:.0f}% cache hits",
            )

    def page_ready(self, page: int, index: int, total_pages: int, wav, duration_s: float, sentences: list) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                description=f"{page}/{total_pages} pages ready",
            )
