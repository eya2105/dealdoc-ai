"""Pure text chunking"""
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _snap_to_word_start(text: str, pos: int) -> int:
    """Move `pos` forward to the first character of the next word"""
    if pos == 0 or text[pos - 1] == " ":
        return pos
    space = text.find(" ", pos)
    return len(text) if space == -1 else space + 1


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks that start and end on word boundaries."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            snapped = text.rfind(" ", start, end)
            if snapped > start:
                end = snapped
        if chunk := text[start:end].strip():
            chunks.append(chunk)
        if end >= n:  # reached the end: no redundant tail chunk
            break
        next_start = _snap_to_word_start(text, max(end - overlap, start + 1))
        start = min(next_start, end)  # always progress, never skip text
    return chunks
