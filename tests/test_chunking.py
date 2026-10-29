import pytest

from chunking import chunk_text


def test_short_text_gives_single_chunk():
    assert chunk_text("hello world", chunk_size=100, overlap=10) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []


def test_no_redundant_tail_chunk():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    # the last chunk must bring new text, not be fully contained in the previous one
    assert chunks[-1] not in chunks[-2]


def test_chunks_respect_size_and_cover_all_words():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), chunk_size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    covered = set(" ".join(chunks).split())
    assert covered == set(words)


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()) & set(b.split())


def test_cuts_happen_on_word_boundaries():
    text = " ".join(f"word{i}" for i in range(200))
    for chunk in chunk_text(text, chunk_size=100, overlap=20):
        assert all(w.startswith("word") for w in chunk.split())


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)
