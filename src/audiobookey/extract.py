"""Page-exact text extraction from PDFs with pymupdf.

P4-A additions: OCR fallback for scanned pages (pages with no text layer).
Scanned pages are rendered with pymupdf's built-in pixmap renderer (no poppler
dependency) and OCRed via pytesseract/tesseract. OCR page text is cached under
``data/ocr-cache`` keyed by (pdf sha256, lang, page) so re-converts skip
re-OCR (~2-5 s per page).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Standard Windows install location, checked before PATH lookup.
TESSERACT_WINDOWS_DIR = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
TESSERACT_INSTALL_HINT = (
    "tesseract missing, install with: winget install UB-Mannheim.Tesseract-OCR"
)

OCR_DPI = 200


@dataclass
class SourcePage:
    number: int  # 1-based PDF page number
    text: str


class TesseractMissingError(RuntimeError):
    """Raised when OCR was requested but no tesseract binary is available."""


def find_tesseract() -> str | None:
    """Locate a tesseract binary: the standard Windows dir first, then PATH."""
    if TESSERACT_WINDOWS_DIR.is_file():
        return str(TESSERACT_WINDOWS_DIR)
    return shutil.which("tesseract")


def ensure_tesseract(tesseract_cmd: str | None = None) -> str:
    """Return a tesseract executable path or raise TesseractMissingError."""
    if tesseract_cmd:
        return tesseract_cmd
    found = find_tesseract()
    if not found:
        raise TesseractMissingError(TESSERACT_INSTALL_HINT)
    return found


def _import_pytesseract():
    """Import pytesseract lazily so --ocr is an optional dependency."""
    try:
        import pytesseract  # noqa: PLC0415 (optional extra: uv sync --extra ocr)
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "pytesseract is not installed; run `uv sync --extra ocr`"
        ) from exc
    return pytesseract


def _bind_tesseract_cmd(pytesseract, cmd: str) -> None:
    """Point pytesseract at our detected binary (best effort for stubs)."""
    engine = getattr(pytesseract, "pytesseract", pytesseract)
    if hasattr(engine, "tesseract_cmd"):
        engine.tesseract_cmd = cmd


def _pdf_sha256(pdf_path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pages_without_text(pdf_path: str | Path) -> list[int]:
    """1-based page numbers whose embedded text layer is empty (scanned)."""
    doc = pymupdf.open(str(pdf_path))
    try:
        return [
            number
            for number, page in enumerate(doc, start=1)
            if not (page.get_text("text") or "").strip()
        ]
    finally:
        doc.close()


def _ocr_cache_path(cache_dir: Path, pdf_sha: str, page: int, lang: str) -> Path:
    return cache_dir / f"{pdf_sha}-{lang}-p{page}.txt"


def ocr_missing_pages(
    pdf_path: str | Path,
    lang: str = "eng",
    cache_dir: str | Path | None = None,
    dpi: int = OCR_DPI,
    tesseract_cmd: str | None = None,
) -> dict[int, str]:
    """OCR every page without a text layer; return {page_number: text}.

    Results are cached under ``cache_dir`` (default ``data/ocr-cache``) keyed
    by (pdf sha256, lang, page), so re-converts skip re-OCR. Pages that
    already have text are never OCRed.
    """
    os_pdf = str(pdf_path)
    missing = pages_without_text(os_pdf)
    if not missing:
        return {}

    cmd = ensure_tesseract(tesseract_cmd)
    if cache_dir is None:
        from .voices import find_data_dir

        cache_dir = find_data_dir() / "ocr-cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdf_sha = _pdf_sha256(os_pdf)
    result: dict[int, str] = {}
    uncached: list[int] = []
    for page in missing:
        cached = _ocr_cache_path(cache_dir, pdf_sha, page, lang)
        if cached.is_file():
            result[page] = cached.read_text(encoding="utf-8")
        else:
            uncached.append(page)

    if not uncached:
        return result

    pytesseract = _import_pytesseract()
    _bind_tesseract_cmd(pytesseract, cmd)
    doc = pymupdf.open(os_pdf)
    try:
        for page in uncached:
            # Render with pymupdf (no poppler), hand the PNG file to tesseract.
            png_bytes = doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")
            tmp_png = cache_dir / f".p{page}.{os.getpid()}.tmp.png"
            tmp_png.write_bytes(png_bytes)
            try:
                text = pytesseract.image_to_string(str(tmp_png), lang=lang)
            finally:
                tmp_png.unlink(missing_ok=True)
            result[page] = text
            _ocr_cache_path(cache_dir, pdf_sha, page, lang).write_text(
                text, encoding="utf-8"
            )
    finally:
        doc.close()
    return result


def ensure_text_layer(
    pdf_path: str | Path,
    out_pdf: str | Path,
    lang: str = "eng",
    cache_dir: str | Path | None = None,
    dpi: int = OCR_DPI,
    tesseract_cmd: str | None = None,
) -> Path:
    """Write a copy of the PDF with invisible OCR text on scanned pages.

    Scanned pages get their OCRed text inserted invisibly (render mode 3), so
    a conversion of the copy extracts text like any born-digital book. Pages
    that already have text are untouched.
    """
    ocr_texts = ocr_missing_pages(
        pdf_path,
        lang=lang,
        cache_dir=cache_dir,
        dpi=dpi,
        tesseract_cmd=tesseract_cmd,
    )
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    try:
        for page_number, text in ocr_texts.items():
            page = doc[page_number - 1]
            page.insert_text(
                (72, 72),
                text,
                fontsize=10,
                render_mode=3,  # invisible glyphs: text layer only
            )
        doc.save(str(out_pdf), garbage=3, deflate=True)
    finally:
        doc.close()
    return out_pdf


def extract_pages(pdf_path: str | Path) -> list[SourcePage]:
    """Return raw text per page, in page order."""
    doc = pymupdf.open(str(pdf_path))
    try:
        pages = [
            SourcePage(number=i, text=page.get_text("text") or "")
            for i, page in enumerate(doc, start=1)
        ]
    finally:
        doc.close()
    return pages
