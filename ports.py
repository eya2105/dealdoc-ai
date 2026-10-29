"""Ports: the small interfaces the application depends on (Dependency Inversion).

Each one is a typing.Protocol, so any object with these methods qualifies. The Chroma store,
the Groq client, the MiniLM embedder and the cross-encoder are only the *current* adapters:
swapping one (a local LLM, another vector DB) touches the adapter and the wiring, not the logic.

A chunk is a plain dict: {"chunk_id", "text", "score", "metadata"}, plus "rerank_score"
once a reranker has scored it. `score` is a similarity (higher is better).
"""
from typing import Optional, Protocol

Chunk = dict


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Chunk], top_k: int) -> list[Chunk]: ...


class LLM(Protocol):
    def complete_text(self, system: str, user: str, temperature: float = 0.1) -> str: ...

    def complete_json(self, system: str, user: str, temperature: float = 0.0) -> Optional[dict]:
        """A parsed JSON object, or None when the call or the parsing failed."""
        ...


class VectorIndex(Protocol):
    """What ingestion and retrieval need from the vector database."""

    def replace_document(self, deal_id: str, doc_id: str, ids: list[str], texts: list[str],
                         embeddings: list[list[float]], metadatas: list[dict]) -> None: ...

    def search(self, query_embeddings: list[list[float]], n_results: int,
               deal_id: Optional[str] = None, doc_id: Optional[str] = None) -> list[list[Chunk]]:
        """One best-first list of chunks per query embedding."""
        ...


class DocumentCatalog(Protocol):
    """What the API needs to know about what is indexed (kept apart from VectorIndex: ISP)."""

    def list_documents(self) -> list[dict]: ...

    def deal_exists(self, deal_id: str) -> bool: ...

    def get_texts(self, ids: list[str]) -> list[str]: ...

    def ping(self) -> None: ...
