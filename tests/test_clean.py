"""Cleanup unit tests: header/footer strip, page numbers, hyphens, de-wrap."""

from audiobookey.clean import (
    clean_pages,
    find_running_lines,
    flow_text,
    normalize_line,
)

PAGES = [
    "My Book\n\nIt was the best of times, it was the worst of times.\n"
    "It was the age of wisdom.\n\n1",
    "My Book\n\nIt was the epoch of belief.\n\nIn short, the period was so\nfar like the present.\n\n2",
    "My Book\n\nThe period is so far like the present age.\n\n3",
]


def test_normalize_line_matches_page_numbers():
    assert normalize_line("Page 12") == normalize_line("page 34")
    assert normalize_line("12") == normalize_line("99")


def test_find_running_lines_detects_common_headers():
    raw = [
        "My Book\nbody text one\n\n1",
        "My Book\nbody text two\n\n2",
        "My Book\nbody text three\n\n3",
    ]
    assert "my book" in find_running_lines(raw)
    assert "body text one" not in find_running_lines(raw)


def test_clean_strips_headers_and_bare_page_numbers():
    result = clean_pages(PAGES)
    for page in result:
        assert page.text, "page unexpectedly empty"
        assert "my book" not in page.text.lower()
        # header body ("My Book") is gone, page body remains
        assert "my book" not in page.text.lower()
        assert len(page.text) > 20
    # no isolated digits at all (page numbers were the only digits)
    assert not any(ch.isdigit() for p in result for ch in p.text)


def test_hyphenated_linebreak_rejoined():
    text = "This is the exam-\nple the reader asked for."
    out = flow_text(text.splitlines())
    assert "example the reader asked for." in out
    assert "exam-\n" not in out
    assert "exam- ple" not in out


def test_dewrap_keeps_paragraph_breaks():
    text = "First paragraph,\nwrapped over lines.\n\nSecond paragraph\nalso wrapped."
    out = flow_text(text.splitlines())
    assert out == (
        "First paragraph, wrapped over lines.\n\nSecond paragraph also wrapped."
    )


def test_pages_without_text_are_skipped_with_text_recorded():
    raw = ["Some words here.\n\n1", "", "More words.\n\n3"]
    result = clean_pages(raw)
    assert result[1].skipped is True
    assert result[1].text == ""
    assert result[0].skipped is False
    assert result[2].skipped is False
