import pytest

from reranker import CrossEncoderReranker
from tests.fakes import write_html


class FakeCrossEncoder:
    """Scores 1.0 for pairs whose chunk mentions 'needle', else 0.0."""

    def predict(self, pairs, **kwargs):
        return [1.0 if "needle" in text else 0.0 for _, text in pairs]


def test_rerank_sorts_by_cross_encoder_score():
    candidates = [{"chunk_id": "a", "text": "hay"}, {"chunk_id": "b", "text": "a needle here"},
                  {"chunk_id": "c", "text": "more hay"}]
    out = CrossEncoderReranker(model=FakeCrossEncoder()).rerank("q", candidates, top_k=2)
    assert out[0]["chunk_id"] == "b" and out[0]["rerank_score"] == 1.0
    assert len(out) == 2


def test_rerank_empty_candidates():
    assert CrossEncoderReranker(model=FakeCrossEncoder()).rerank("q", [], top_k=5) == []


def test_rerank_does_not_mutate_input():
    candidates = [{"chunk_id": "a", "text": "needle"}]
    CrossEncoderReranker(model=FakeCrossEncoder()).rerank("q", candidates, top_k=1)
    assert "rerank_score" not in candidates[0]


@pytest.fixture
def needle_doc(container, tmp_path):
    body = " ".join(f"filler{i}" for i in range(700)) + " the needle is hidden in the last chunk"
    container.ingestor.ingest(write_html(tmp_path / "d.html", body), deal_id="d", doc_id="doc")


def test_retrieve_with_rerank_puts_the_reranker_winner_first(container, needle_doc):
    top = container.retriever.retrieve("anything", top_k=1, deal_id="d", rerank=True)[0]
    assert "needle" in top["text"] and top["rerank_score"] == 1.0
    assert "score" in top  # the cosine score is kept


def test_retrieve_without_rerank_has_no_rerank_score(container, needle_doc):
    results = container.retriever.retrieve("anything", top_k=3, deal_id="d", rerank=False)
    assert all("rerank_score" not in r for r in results)


def test_rerank_default_follows_the_constructor_flag(container, needle_doc, store, embedder, reranker, llm):
    from retriever import Retriever

    on = Retriever(store, embedder, llm, reranker, rerank_enabled=True, expansion_enabled=False)
    off = Retriever(store, embedder, llm, reranker, rerank_enabled=False, expansion_enabled=False)
    assert "rerank_score" in on.retrieve("anything", top_k=2, deal_id="d")[0]
    assert "rerank_score" not in off.retrieve("anything", top_k=2, deal_id="d")[0]
