"""Chunking unit tests: sentence boundaries and the phoneme cap."""

from audiobookey.chunk import (
    MAX_PHONEMES,
    PhonemeCounter,
    chunk_text,
    split_sentences,
)


def test_split_sentences_basic():
    text = "One sentence here. Another follows! And a third? Yes."
    assert split_sentences(text) == [
        "One sentence here.",
        "Another follows!",
        "And a third?",
        "Yes.",
    ]


def test_split_sentences_respects_abbreviations_and_initials():
    text = "Dr. Smith met Mr. Jones. They talked to J. K. Rowling at 5 p. m."
    parts = split_sentences(text)
    # abbreviation periods do not end a sentence; the real stop after 'Jones.' does
    assert parts[0] == "Dr. Smith met Mr. Jones."


def test_counting_fallback_is_deterministic():
    count = PhonemeCounter()
    n1 = count("hello world")
    n2 = count("hello world")
    assert n1 == n2 > 0


def test_chunking_respects_cap_with_fallback_counter():
    count = PhonemeCounter()
    sentence = "The quick brown fox jumps over the lazy dog. " * 30
    chunks = chunk_text(sentence.strip(), count=count)
    for chunk in chunks:
        assert count(chunk) <= MAX_PHONEMES
    # text preserved (modulo whitespace)
    assert " ".join(chunks).split() == sentence.split()


def test_chunking_never_splits_mid_sentence_when_it_fits():
    text = "First short sentence. Second short sentence. Third short sentence."
    count = PhonemeCounter()
    chunks = chunk_text(text, count=count)
    assert chunks == [text]


def test_chunking_splits_sentences_never_mid_word():
    text = " ".join(f"Sentence number {i} is quite ordinary and plain." for i in range(40))
    count = PhonemeCounter()
    chunks = chunk_text(text, count=count)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count(chunk) <= MAX_PHONEMES
    assert " ".join(chunks).split() == text.split()


def test_huge_single_sentence_is_split_at_word_boundaries():
    text = "word " * 2000
    count = PhonemeCounter()
    chunks = chunk_text(text.strip(), count=count)
    assert all(count(c) <= MAX_PHONEMES for c in chunks)
    assert len(chunks) > 1
    assert " ".join(chunks).split() == text.split()
