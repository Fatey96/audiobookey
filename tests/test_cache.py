"""Cache keying and behaviour tests."""

from pathlib import Path

import numpy as np

from audiobookey.audio import write_wav
from audiobookey.cache import ChunkCache, book_id, chunk_key


def test_chunk_key_changes_with_every_component():
    base = chunk_key("some text", "af_heart", 1.0, "kokoro-onnx==0.6.1;ep=cuda")
    assert chunk_key("other text", "af_heart", 1.0, "kokoro-onnx==0.6.1;ep=cuda") != base
    assert chunk_key("some text", "am_michael", 1.0, "kokoro-onnx==0.6.1;ep=cuda") != base
    assert chunk_key("some text", "af_heart", 1.1, "kokoro-onnx==0.6.1;ep=cuda") != base
    assert chunk_key("some text", "af_heart", 1.0, "kokoro-onnx==0.6.1;ep=cpu") != base


def test_chunk_key_extra_changes_key():
    base = chunk_key("t", "fp", 1.0, "engine")
    assert chunk_key("t", "fp", 1.0, "engine", extra="0.3") != base
    assert chunk_key("t", "fp", 1.0, "engine", extra="0.3") == chunk_key(
        "t", "fp", 1.0, "engine", extra="0.3"
    )


def test_chunk_key_is_stable_and_a_valid_sha256():
    a = chunk_key("some text", "af_heart", 1.0, "engine")
    b = chunk_key("some text", "af_heart", 1.0, "engine")
    assert a == b
    assert len(a) == 64
    int(a, 16)  # hex


def test_cache_get_store_roundtrip(tmp_path: Path):
    cache = ChunkCache(tmp_path / "chunks")
    key = chunk_key("text", "af_heart", 1.0, "engine")
    assert cache.get(key) is None
    assert cache.misses == 1 and cache.hits == 0

    src = tmp_path / "src.wav"
    write_wav(src, np.zeros(2400, dtype=np.float32), 24000)
    cache.store(key, src)

    got = cache.get(key)
    assert got == tmp_path / "chunks" / f"{key}.wav"
    assert got.is_file()
    assert cache.hits == 1 and cache.misses == 1
    assert not src.exists(), "source wav should be moved into the cache"


def test_cache_rejects_header_only_wav(tmp_path: Path):
    cache = ChunkCache(tmp_path / "chunks")
    key = chunk_key("text", "af_heart", 1.0, "engine")
    (tmp_path / "chunks" / f"{key}.wav").write_bytes(b"RIFF" + b"\x00" * 40)
    assert cache.get(key) is None  # truncated write treated as a miss


def test_book_id_is_content_based(tmp_path: Path):
    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.pdf"  # same content, different name
    p3 = tmp_path / "c.pdf"
    for p in (p1, p2):
        p.write_bytes(b"%PDF-1.4 same content")
    p3.write_bytes(b"%PDF-1.4 different")
    assert book_id(p1) == book_id(p2)
    assert book_id(p1) != book_id(p3)
