"""Composition root: the one place that knows which concrete classes implement the ports.

Everything else receives its collaborators through its constructor. Tests call
`build_container(...)` with fakes; production calls `get_container()`.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from embeddings import SentenceTransformerEmbedder
from extraction import Extractor
from generator import AnswerGenerator
from ingest import Ingestor
from llm import GroqLLM
from ports import LLM, Embedder, Reranker
from reranker import CrossEncoderReranker
from retriever import Retriever
from service import RagService
from store import ChromaStore


@dataclass
class Container:
    store: ChromaStore
    embedder: Embedder
    reranker: Reranker
    llm: LLM
    retriever: Retriever
    generator: AnswerGenerator
    extractor: Extractor
    ingestor: Ingestor
    service: RagService


def build_container(*, store=None, embedder=None, reranker=None, llm=None,
                    retriever_options: Optional[dict] = None,
                    extractor_options: Optional[dict] = None) -> Container:
    """Wire the application. Any adapter can be replaced; options go to Retriever / Extractor."""
    store = store or ChromaStore()
    embedder = embedder or SentenceTransformerEmbedder()
    reranker = reranker or CrossEncoderReranker()
    llm = llm or GroqLLM()

    retriever = Retriever(store, embedder, llm, reranker, **(retriever_options or {}))
    generator = AnswerGenerator(llm)
    extractor = Extractor(retriever, llm, **(extractor_options or {}))
    ingestor = Ingestor(store, embedder)
    service = RagService(ingestor, retriever, generator, extractor, catalog=store)
    return Container(store, embedder, reranker, llm, retriever, generator, extractor, ingestor, service)


@lru_cache(maxsize=1)
def get_container() -> Container:
    return build_container()


def get_rag_service() -> RagService:
    """FastAPI dependency (tests replace it with app.dependency_overrides)."""
    return get_container().service
