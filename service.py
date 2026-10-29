"""Application service: the use cases behind the API, free of HTTP and of any concrete library."""
from config import TOP_K
from ports import DocumentCatalog


class DealNotFoundError(LookupError):
    """No indexed document belongs to this deal."""


class RagService:
    """Facade over ingestion, question answering, extraction and the catalog.

    Routes stay thin (parse the request, map errors to status codes) and all the
    behaviour lives here, where it can be tested without a web server.
    """

    def __init__(self, ingestor, retriever, generator, extractor, catalog: DocumentCatalog,
                 preview_chars: int = 150):
        self._ingestor, self._retriever, self._generator = ingestor, retriever, generator
        self._extractor, self._catalog, self._preview_chars = extractor, catalog, preview_chars

    def ingest(self, file_path: str, deal_id=None, doc_id=None, doc_type: str = "unknown") -> dict:
        return self._ingestor.ingest(file_path, deal_id=deal_id, doc_id=doc_id, doc_type=doc_type)

    def ask(self, question: str, top_k: int = TOP_K, deal_id=None, doc_id=None) -> dict:
        chunks = self._retriever.retrieve(question, top_k=top_k, deal_id=deal_id, doc_id=doc_id)
        if not chunks:
            return {"answer": "No relevant chunks found.", "sources": []}
        answer = self._generator.answer(question, [c["text"] for c in chunks])
        sources = [
            {
                "rank": rank,  # matches the [Source N] citations in the answer
                "chunk_id": c["chunk_id"],
                "deal_id": c["metadata"]["deal_id"],
                "doc_id": c["metadata"]["doc_id"],
                "source_file": c["metadata"].get("source_file", ""),
                "score": round(c["score"], 4),
                "text": c["text"],  # the full passage, so the UI can show what the answer is based on
                "text_preview": c["text"][:self._preview_chars] + "...",
            }
            for rank, c in enumerate(chunks, start=1)
        ]
        return {"answer": answer, "sources": sources}

    def extract(self, deal_id: str, fields: list[str]) -> dict[str, dict]:
        if not self._catalog.deal_exists(deal_id):
            raise DealNotFoundError(f"Deal '{deal_id}' not found")
        return self._extractor.extract(deal_id, fields)  # searches ALL documents of the deal

    def documents(self) -> list[dict]:
        return self._catalog.list_documents()

    def check_health(self) -> None:
        """Raises if the index is unreachable."""
        self._catalog.ping()