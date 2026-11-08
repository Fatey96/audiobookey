"""Lexicon unit tests: parsing, string/regex rules, cache-relevant behaviour."""

from __future__ import annotations

import pytest

from audiobookey.lexicon import Lexicon, LexiconError, load_lexicon


def test_string_rule_replaces_literal(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("doughnut => donut\n", encoding="utf-8")
    lex = load_lexicon(p)
    assert lex.apply("A doughnut and another doughnut.") == (
        "A donut and another donut."
    )


def test_regex_rule_with_r_prefix(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text(r"r:[Cc]o\s?\d+\b => Company" + "\n", encoding="utf-8")
    lex = load_lexicon(p)
    assert lex.apply("See Co 7 and co9.") == "See Company and Company."
    # replacement stays literal: no backslash escape interpretation
    p2 = tmp_path / "lex2.txt"
    p2.write_text(r"r:\bDot\b => N\A" + "\n", encoding="utf-8")
    assert load_lexicon(p2).apply("Dot") == "N\\A"


def test_comments_and_blank_lines_skipped(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text(
        "# a comment\n\n   \nalpha => beta\n", encoding="utf-8"
    )
    lex = load_lexicon(p)
    assert len(lex.rules) == 1


def test_rules_apply_in_file_order(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("cat -> dog\ndog -> bird\n", encoding="utf-8")
    lex = load_lexicon(p)
    # 'cat' -> 'dog' -> 'bird' (second rule re-matches the first's output)
    assert lex.apply("cat") == "bird"


def test_missing_separator_raises(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("no separator here\n", encoding="utf-8")
    with pytest.raises(LexiconError):
        load_lexicon(p)


def test_invalid_regex_raises(tmp_path):
    p = tmp_path / "lex.txt"
    p.write_text("r:([unclosed => x\n", encoding="utf-8")
    with pytest.raises(LexiconError):
        load_lexicon(p)


def test_lexicon_changes_text_so_cache_key_changes():
    from audiobookey.cache import chunk_key

    raw = "The dragon flew over doughnut hill."
    lex = Lexicon([])
    bare = lex.apply(raw)

    keyed = Lexicon([])
    from audiobookey.lexicon import LexiconRule

    keyed = Lexicon([LexiconRule("doughnut", "pie", False, 1)])
    changed = keyed.apply(raw)

    k1 = chunk_key(bare, "fp", 1.0, "engine")
    k2 = chunk_key(changed, "fp", 1.0, "engine")
    assert changed != bare
    assert k1 != k2, "lexicon-changed text must invalidate the chunk cache"
