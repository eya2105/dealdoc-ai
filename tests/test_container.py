import config
from container import build_container
from extraction import Extractor
from generator import AnswerGenerator
from ingest import Ingestor
from retriever import Retriever
from service import RagService
from store import ChromaStore
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker


def test_default_wiring_needs_no_model_no_key_and_no_disk(tmp_path):
    """Everything heavy is lazy: building the application is cheap and side-effect free."""
    c = build_container(store=ChromaStore(path=str(tmp_path / "never-created")))
    assert isinstance(c.service, RagService) and isinstance(c.retriever, Retriever)
    assert not (tmp_path / "never-created").exists()


def test_defaults_come_from_config(store):
    c = build_container(store=store)
    assert c.retriever._rerank_enabled == config.RERANK_ENABLED
    assert c.retriever._expansion_enabled == config.QUERY_EXPANSION_ENABLED
    assert c.extractor._chunks_per_field == config.EXTRACT_CHUNKS_PER_FIELD


def test_collaborators_are_shared_and_replaceable(store):
    embedder, llm, reranker = FakeEmbedder(), FakeLLM(), FakeReranker()
    c = build_container(store=store, embedder=embedder, llm=llm, reranker=reranker,
                        extractor_options={"chunks_per_field": 2})
    assert c.generator._llm is llm and c.extractor._llm is llm and c.retriever._llm is llm
    assert c.ingestor._embedder is embedder and c.retriever._embedder is embedder
    assert c.ingestor._index is store and c.retriever._index is store
    assert c.extractor._retriever is c.retriever
    assert c.extractor._chunks_per_field == 2
    assert isinstance(c.ingestor, Ingestor) and isinstance(c.generator, AnswerGenerator)
    assert isinstance(c.extractor, Extractor)
