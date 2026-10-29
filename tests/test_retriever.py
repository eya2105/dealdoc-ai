from retriever import Retriever
from tests.fakes import FakeLLM, write_html


def test_expand_query_keeps_only_non_empty_strings_and_at_most_three(store, embedder, reranker):
    llm = FakeLLM(json_fn=lambda s, u: {"queries": ["a", "", 42, "b", "c", "d"]})
    assert Retriever(store, embedder, llm, reranker).expand_query("q") == ["a", "b", "c"]


def test_expand_query_returns_nothing_when_the_llm_fails(store, embedder, reranker):
    assert Retriever(store, embedder, FakeLLM(json_fn=lambda s, u: None), reranker).expand_query("q") == []


def test_expansion_adds_variant_queries_to_the_search(container, tmp_path):
    container.ingestor.ingest(write_html(tmp_path / "a.html", "alpha text about prices"), deal_id="d", doc_id="a")
    container.ingestor.ingest(write_html(tmp_path / "b.html", "beta text about dates"), deal_id="d", doc_id="b")
    container.llm.json_fn = lambda s, u: {"queries": ["beta text about dates"]}
    without = container.retriever.retrieve("alpha text about prices", top_k=1, deal_id="d", expand=False)
    with_variants = container.retriever.retrieve("alpha text about prices", top_k=2, deal_id="d", expand=True)
    assert len(container.llm.json_calls) == 1  # one expansion call, and only for expand=True
    assert {c["metadata"]["doc_id"] for c in with_variants} == {"a", "b"}
    assert without[0]["metadata"]["doc_id"] == "a"


def test_top_k_is_clamped(container, tmp_path):
    body = " ".join(f"w{i}" for i in range(6000))
    container.ingestor.ingest(write_html(tmp_path / "a.html", body), deal_id="d", doc_id="a")
    assert len(container.retriever.retrieve("w1", top_k=10_000, deal_id="d")) == 20  # MAX_TOP_K
    assert len(container.retriever.retrieve("w1", top_k=0, deal_id="d")) == 1


def test_duplicates_across_queries_keep_the_best_score(embedder, reranker, llm):
    class TwoQueryIndex:
        def search(self, query_embeddings, n_results, deal_id=None, doc_id=None):
            low = {"chunk_id": "x", "text": "t", "score": 0.2, "metadata": {}}
            high = {"chunk_id": "x", "text": "t", "score": 0.9, "metadata": {}}
            return [[low], [high]]

    retriever = Retriever(TwoQueryIndex(), embedder, llm, reranker)
    llm.json_fn = lambda s, u: {"queries": ["second"]}
    result = retriever.retrieve("q", expand=True, rerank=False)
    assert len(result) == 1 and result[0]["score"] == 0.9
