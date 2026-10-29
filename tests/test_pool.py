import pytest

from config import CANDIDATES_PER_QUERY
from tests.fakes import write_html


@pytest.fixture
def many_chunks(container, tmp_path):
    body = " ".join(f"word{i}" for i in range(6000))  # ~50k chars -> ~60 chunks
    f = write_html(tmp_path / "d.html", body)
    assert container.ingestor.ingest(f, deal_id="d", doc_id="doc")["total_chunks"] > 40


def test_default_pool_is_the_configured_candidate_count(container, many_chunks):
    results = container.retriever.retrieve("word5", top_k=20, deal_id="d")
    assert len(results) == min(20, CANDIDATES_PER_QUERY)


def test_pool_limits_the_candidates(container, many_chunks):
    assert len(container.retriever.retrieve("word5", top_k=20, deal_id="d", pool=5)) == 5


def test_reranker_only_sees_pool_candidates(container, many_chunks, reranker):
    container.retriever.retrieve("word5", top_k=3, deal_id="d", rerank=True, pool=12)
    container.retriever.retrieve("word5", top_k=3, deal_id="d", rerank=True)
    assert reranker.pool_sizes == [12, CANDIDATES_PER_QUERY]
