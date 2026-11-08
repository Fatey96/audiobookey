"""Cover rendering: page 1 of the book's PDF as a 300 px wide PNG."""

from __future__ import annotations

from pathlib import Path


def cover_png(
    pdf_path: str | Path,
    out_path: str | Path,
    width: int = 300,
) -> Path:
    """Render page 1 of *pdf_path* to *out_path* (PNG), cached on disk.

    Re-renders only when the cache file is missing or older than the PDF.
    """
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and out_path.stat().st_mtime >= pdf_path.stat().st_mtime:
        return out_path

    import pymupdf

    doc = pymupdf.open(pdf_path)
    try:
        page = doc.load_page(0)
        zoom = width / page.rect.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        pix.save(str(out_path))
    finally:
        doc.close()
    return out_path
