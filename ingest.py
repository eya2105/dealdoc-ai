"""Ingestion: parse -> chunk -> embed -> store."""
from pathlib import Path
from typing import Callable, Optional

from chunking import chunk_text
from config import PRELOAD_DOCUMENTS
from ids import make_chunk_id, slugify
from parsing import parse_document
from ports import Embedder, VectorIndex


class Ingestor:
    """The parser and the chunking strategy are injected: another file format or another
    way of cutting text is a new function passed in, not an edit of this class."""

    def __init__(self, index: VectorIndex, embedder: Embedder, *,
                 parser: Callable[[str], str] = parse_document,
                 chunker: Callable[[str], list[str]] = chunk_text):
        self._index, self._embedder, self._parser, self._chunker = index, embedder, parser, chunker

    def ingest(self, file_path: str,
               deal_id: Optional[str] = None,
               doc_id: Optional[str] = None,
               doc_type: str = "unknown") -> dict:
        """Index one document. Re-ingesting the same (deal_id, doc_id) replaces it."""
        doc_id = slugify(doc_id or Path(file_path).stem)
        deal_id = slugify(deal_id or doc_id)

        text = self._parser(file_path)
        chunks = self._chunker(text)
        if not chunks:
            raise ValueError("No text could be extracted from this document")
        embeddings = self._embedder.embed(chunks)  # embed first: a failure must not wipe the old index

        base_metadata = {
            "deal_id": deal_id,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "source_file": Path(file_path).name,
        }
        self._index.replace_document(
            deal_id, doc_id,
            ids=[make_chunk_id(deal_id, doc_id, i) for i in range(len(chunks))],
            texts=chunks,
            embeddings=embeddings,
            metadatas=[{**base_metadata, "chunk_index": i} for i in range(len(chunks))],
        )
        return {
            "deal_id": deal_id,
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "document_chars": len(text),
        }


if __name__ == "__main__":
    from container import get_container

    ingestor = get_container().ingestor
    for doc in PRELOAD_DOCUMENTS:
        result = ingestor.ingest(**doc)
        print(f"Ingested {result['total_chunks']} chunks for {result['deal_id']} / {result['doc_id']}")
