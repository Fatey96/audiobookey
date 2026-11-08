"""P4-A tests: OCR for scanned PDFs + the batch convert queue (backend only).

OCR: synthetic scanned PDFs (text rendered into an image, no text layer) built
in-test with pymupdf. If a real tesseract binary exists on this box the real
OCR path is asserted; otherwise the no-text-page path is covered with a
stubbed pytesseract module.

Batch: server tests convert 2 tiny synthetic PDFs in ONE batch job and assert
a single job id, both books registered and the aggregate counters.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from audiobookey import extract
from audiobookey.extract import (
    TesseractMissingError,
    ensure_text_layer,
    find_tesseract,
    ocr_missing_pages,
    pages_without_text,
)

TESSERACT_PRESENT = find_tesseract() is not None
PYTESSERACT_PRESENT = importlib.util.find_spec("pytesseract") is not None


# -- synthetic PDFs --------------------------------------------------------------


def make_text_pdf(path: str | Path, text: str, pages: int = 1) -> Path:
    """A born-digital PDF with a real text layer."""
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()
    return Path(path)


def make_scanned_pdf(path: str | Path, text: str) -> Path:
    """A 'scanned' PDF: the text is rendered into an image, no text layer."""
    png_path = str(path) + ".src.png"
    tmp_doc = pymupdf.open()
    page = tmp_doc.new_page()
    page.insert_text((72, 72), text, fontsize=14)
    page.get_pixmap(dpi=120).save(png_path)
    tmp_doc.close()

    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), filename=png_path)
    doc.save(str(path))
    doc.close()
    Path(png_path).unlink(missing_ok=True)
    return Path(path)


class FakeTesseract:
    """Stands in for the pytesseract module in stubbed OCR tests."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls = 0
        self.langs: list[str] = []

    def image_to_string(self, image, lang=None, config=None):  # noqa: ANN001
        self.calls += 1
        self.langs.append(lang)
        return self.texts[min(self.calls - 1, len(self.texts) - 1)]


@pytest.fixture()
def stubbed_tesseract(monkeypatch):
    fake = FakeTesseract(["Sample scanned page number 4."])
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    monkeypatch.setattr(extract, "find_tesseract", lambda: "C:/fake/tesseract.exe")
    return fake


# -- tesseract detection --------------------------------------------------------


def test_find_tesseract_reports_missing_or_present():
    # The dev box under test has no tesseract (see P4-A run notes): the
    # detection contract must hold either way and missing must produce the
    # documented install hint on use.
    found = find_tesseract()
    assert found is None or Path(found).is_file()


def test_missing_tesseract_fails_with_install_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(extract, "find_tesseract", lambda: None)
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "Scanned words needing OCR.")
    with pytest.raises(TesseractMissingError) as excinfo:
        ocr_missing_pages(pdf, cache_dir=tmp_path / "cache")
    assert "winget install UB-Mannheim.Tesseract-OCR" in str(excinfo.value)


# -- page-level OCR ------------------------------------------------------------


def test_scanned_pdf_has_no_text_layer(tmp_path):
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "Scanned page words.")
    assert pages_without_text(pdf) == [1]


def test_text_pdf_has_no_missing_pages(tmp_path):
    pdf = make_text_pdf(tmp_path / "text.pdf", "Born digital page.")
    assert pages_without_text(pdf) == []


def test_scanned_plus_text_pdf_reports_only_scanned_pages(tmp_path):
    # render the source page to an image (like a scan), then build a mixed doc
    src = make_text_pdf(tmp_path / "src.pdf", "Some rendered content.")
    doc_src = pymupdf.open(str(src))
    png_path = tmp_path / "page-img.png"
    doc_src[0].get_pixmap(dpi=100).save(str(png_path))
    doc_src.close()

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "born digital", fontsize=12)
    other = doc.new_page()
    other.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(png_path))
    pdf = tmp_path / "mixed.pdf"
    doc.save(str(pdf))
    doc.close()
    png_path.unlink(missing_ok=True)
    assert pages_without_text(pdf) == [2]


def test_ocr_missing_pages_with_stubbed_tesseract(tmp_path, stubbed_tesseract):
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "image only")
    cache = tmp_path / "ocr-cache"
    texts = ocr_missing_pages(pdf, cache_dir=cache)

    assert texts == {1: "Sample scanned page number 4."}
    assert stubbed_tesseract.calls == 1
    assert stubbed_tesseract.langs == ["eng"]  # --ocr-lang reaches tesseract
    cached = list(cache.glob("*.txt"))
    assert len(cached) == 1


def test_ocr_cache_prevents_re_ocr(tmp_path, stubbed_tesseract):
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "image only")
    cache = tmp_path / "cache"
    first = ocr_missing_pages(pdf, cache_dir=cache)
    assert first == {1: "Sample scanned page number 4."}
    assert stubbed_tesseract.calls == 1

    # Same pdf again: served from the cache, tesseract untouched.
    second = ocr_missing_pages(pdf, cache_dir=cache)
    assert second == first
    assert stubbed_tesseract.calls == 1


def test_ocr_cache_keyed_by_pdf_content(tmp_path, monkeypatch):
    fake = FakeTesseract(["alpha run", "beta run"])
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    monkeypatch.setattr(extract, "find_tesseract", lambda: "C:/fake/tesseract.exe")

    pdf_a = make_scanned_pdf(tmp_path / "a.pdf", "image one")
    pdf_b = make_scanned_pdf(tmp_path / "b.pdf", "image two")
    cache = tmp_path / "cache"
    assert ocr_missing_pages(pdf_a, cache_dir=cache) == {1: "alpha run"}
    assert ocr_missing_pages(pdf_b, cache_dir=cache) == {1: "beta run"}
    assert fake.calls == 2  # different sha256 -> cache miss -> re-OCR


def test_ocr_lang_passed_through(tmp_path, stubbed_tesseract):
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "image only")
    ocr_missing_pages(pdf, lang="deu", cache_dir=tmp_path / "cache")
    assert stubbed_tesseract.langs == ["deu"]


def test_ensure_text_layer_makes_scanned_pages_extractable(
    tmp_path, stubbed_tesseract
):
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "image only")
    out = ensure_text_layer(
        pdf, tmp_path / "scan-ocr.pdf", cache_dir=tmp_path / "cache"
    )
    assert out.is_file()
    assert pages_without_text(out) == []
    pages = extract.extract_pages(out)
    assert "Sample scanned page number 4." in pages[0].text


def test_ensure_text_layer_single_file_cli_flag_defaults(tmp_path, stubbed_tesseract):
    # The CLI's --ocr flow: an OCRed copy replaces the scanned input.
    from audiobookey.cli import build_parser

    args = build_parser().parse_args(["convert", "a.pdf", "--ocr"])
    assert args.pdf == ["a.pdf"]
    assert args.ocr is True
    assert args.ocr_lang == "eng"


def test_cli_parser_accepts_multiple_pdfs_and_ocr_flags():
    from audiobookey.cli import build_parser

    args = build_parser().parse_args(
        ["convert", "a.pdf", "b.pdf", "--ocr", "--ocr-lang", "deu"]
    )
    assert args.pdf == ["a.pdf", "b.pdf"]
    assert args.ocr is True
    assert args.ocr_lang == "deu"


@pytest.mark.skipif(
    not (TESSERACT_PRESENT and PYTESSERACT_PRESENT),
    reason="tesseract binary or pytesseract not available on this box",
)
def test_real_ocr_on_scanned_pdf(tmp_path):
    pdf = make_scanned_pdf(tmp_path / "scan.pdf", "Sample scanned page number 4")
    assert pages_without_text(pdf) == [1]
    texts = ocr_missing_pages(pdf, cache_dir=tmp_path / "ocr")
    text = texts[1].lower()
    assert "sample" in text or ("scanned" in text and "page" in text)


# -- batch convert queue (server) ----------------------------------------------


def make_client() -> TestClient:
    from audiobookey.server.api import create_app

    return TestClient(create_app())


def wait_job_done(client: TestClient, job_id: int, timeout: int = 180) -> dict:
    deadline = time.time() + timeout
    job = client.get(f"/api/jobs/{job_id}").json()
    while time.time() < deadline:
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.4)
        job = client.get(f"/api/jobs/{job_id}").json()
    return job


@pytest.fixture()
def server_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"

    import os

    monkeypatch.setenv("AUDIOBOOKEY_DATA_DIR", str(data_dir))
    monkeypatch.chdir(tmp_path)  # keep find_data_dir() away from the repo data/

    pdf_a = make_text_pdf(tmp_path / "book-a.pdf", "Book number one text.")
    pdf_b = make_text_pdf(tmp_path / "book-b.pdf", "Book number two text.")
    return {"data_dir": data_dir, "pdf_a": pdf_a, "pdf_b": pdf_b}


def test_batch_convert_two_pdfs_one_job(server_env):
    with make_client() as client:
        with open(server_env["pdf_a"], "rb") as fa, open(
            server_env["pdf_b"], "rb") as fb:
            resp = client.post(
                "/api/convert/batch",
                files=[
                    ("pdf", ("book-a.pdf", fa, "application/pdf")),
                    ("pdf", ("book-b.pdf", fb, "application/pdf")),
                ],
            )
        assert resp.status_code == 201, resp.text
        initial = resp.json()
        job_id = initial["id"]
        assert "book-a" in initial["message"]
        assert "book-b" in initial["message"]

        job = wait_job_done(client, job_id)
        assert job["status"] == "done", job["message"]

        # ONE batch job only
        assert len(client.get("/api/jobs").json()) == 1

        # both books registered
        books = client.get("/api/books").json()
        assert len(books) == 2
        titles = {b["title"] for b in books}
        assert titles == {"book-a", "book-b"}
        assert job["book_ids"] and len(job["book_ids"]) == 2

        # aggregate counters across the two files (one page each)
        assert job["page_count"] == 2
        assert job["chunk_total"] >= 2
        assert job["chunk_total"] >= job["chunk_done"] >= 1

        # message lists the filenames
        assert "book-a" in job["message"]
        assert "book-b" in job["message"]


def test_batch_json_pdf_paths(server_env):
    with make_client() as client:
        resp = client.post(
            "/api/convert/batch",
            json={
                "pdf_paths": [str(server_env["pdf_a"]), str(server_env["pdf_b"])],
                "ocr": False,
            },
        )
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]

        job = wait_job_done(client, job_id)
        assert job["status"] == "done", job["message"]
        books = client.get("/api/books").json()
        assert len(books) == 2
        assert job["page_count"] == 2


def test_batch_rejects_bad_input(server_env):
    with make_client() as client:
        resp = client.post("/api/convert/batch", json={"pdf_paths": []})
        assert resp.status_code == 400
        assert "pdf_paths" in resp.json()["error"]

        resp = client.post(
            "/api/convert/batch", json={"pdf_paths": ["/nonexistent/nope.pdf"]}
        )
        assert resp.status_code == 404

        missing = server_env["pdf_a"].parent / "wrong.txt"
        missing.write_text("not a pdf")
        resp = client.post(
            "/api/convert/batch", json={"pdf_paths": [str(missing)]}
        )
        assert resp.status_code == 400

        # multipart without any 'pdf' field
        resp = client.post("/api/convert/batch", data={"unrelated": "1"})
        assert resp.status_code == 400


def test_batch_single_entry_stays_single_file_shape(server_env):
    """A one-entry batch must not change the P3 single-file job semantics."""
    with make_client() as client:
        resp = client.post(
            "/api/convert/batch",
            json={"pdf_paths": [str(server_env["pdf_a"])]},
        )
        assert resp.status_code == 201, resp.text
        job_id = resp.json()["id"]

        job = wait_job_done(client, job_id)
        assert job["status"] == "done", job["message"]
        assert "done" in job["message"]
        assert job["page_count"] == 1
        books = client.get("/api/books").json()
        assert len(books) == 1
