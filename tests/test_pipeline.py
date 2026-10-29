"""Ingestor + ChromaStore + Retriever together (real Chroma on a temp dir, fake embeddings)."""
import pytest

from tests.fakes import write_html

LONG_A = " ".join(f"alpha{i}" for i in range(600))
LONG_B = " ".join(f"beta{i}" for i in range(600))


def test_reingest_replaces_instead_of_duplicating(container, tmp_path):
    f = write_html(tmp_path / "a.html", LONG_A)
    first = container.ingestor.ingest(f, deal_id="Deal One", doc_id="Doc A")
    second = container.ingestor.ingest(f, deal_id="Deal One", doc_id="Doc A")
    assert first["total_chunks"] == second["total_chunks"]
    assert container.store.collection.count() == first["total_chunks"]


def test_reingest_shorter_document_leaves_no_stale_chunks(container, tmp_path):
    long_result = container.ingestor.ingest(write_html(tmp_path / "a.html", LONG_A), deal_id="d", doc_id="a")
    short_result = container.ingestor.ingest(write_html(tmp_path / "a.html", "short text"), deal_id="d", doc_id="a")
    assert short_result["total_chunks"] < long_result["total_chunks"]
    assert container.store.collection.count() == short_result["total_chunks"]


def test_two_docs_one_deal_are_listed_and_filterable(container, tmp_path):
    ing = container.ingestor
    ing.ingest(write_html(tmp_path / "a.html", LONG_A), deal_id="d1", doc_id="a", doc_type="merger_agreement")
    ing.ingest(write_html(tmp_path / "b.html", LONG_B), deal_id="d1", doc_id="b", doc_type="proxy_statement")
    ing.ingest(write_html(tmp_path / "c.html", LONG_B), deal_id="d2", doc_id="a")  # same doc_id, other deal

    docs = container.store.list_documents()
    assert [(d["deal_id"], d["doc_id"]) for d in docs] == [("d1", "a"), ("d1", "b"), ("d2", "a")]
    assert all(d["chunks"] > 0 for d in docs)

    whole_deal = container.retriever.retrieve("alpha1", top_k=20, deal_id="d1")
    assert {r["metadata"]["deal_id"] for r in whole_deal} == {"d1"}
    assert {r["metadata"]["doc_id"] for r in whole_deal} == {"a", "b"}

    one_doc = container.retriever.retrieve("alpha1", top_k=20, deal_id="d1", doc_id="b")
    assert {r["metadata"]["doc_id"] for r in one_doc} == {"b"}


def test_same_doc_id_in_two_deals_do_not_collide(container, tmp_path):
    container.ingestor.ingest(write_html(tmp_path / "a.html", LONG_A), deal_id="d1", doc_id="a")
    n1 = container.store.collection.count()
    container.ingestor.ingest(write_html(tmp_path / "b.html", LONG_B), deal_id="d2", doc_id="a")
    assert container.store.collection.count() > n1


def test_deal_exists_and_get_texts(container, tmp_path):
    container.ingestor.ingest(write_html(tmp_path / "a.html", LONG_A), deal_id="d1", doc_id="a")
    assert container.store.deal_exists("d1")
    assert not container.store.deal_exists("nope")
    assert "alpha0" in container.store.get_texts(["d1__a__0"])[0]
    assert container.store.get_texts([]) == []


def test_scores_are_cosine_similarities(container, tmp_path):
    container.ingestor.ingest(write_html(tmp_path / "a.html", "exact phrase here"), deal_id="d", doc_id="a")
    top = container.retriever.retrieve("exact phrase here", top_k=1, deal_id="d")[0]
    assert top["score"] == pytest.approx(1.0, abs=1e-3)  # identical text -> cosine 1.0


def test_empty_document_is_rejected(container, tmp_path):
    f = tmp_path / "empty.html"
    f.write_text("<html><body></body></html>", encoding="utf-8")
    with pytest.raises(ValueError):
        container.ingestor.ingest(str(f), deal_id="d", doc_id="a")


def test_ingestor_accepts_other_parsers_and_chunkers(store, embedder, tmp_path):
    """Strategy pattern: new file formats / chunking strategies are injected, not coded into Ingestor."""
    from ingest import Ingestor

    ingestor = Ingestor(store, embedder, parser=lambda path: "one two three four",
                        chunker=lambda text: text.split())
    result = ingestor.ingest(str(tmp_path / "whatever.html"), deal_id="d", doc_id="a")
    assert result["total_chunks"] == 4
    assert store.get_texts(["d__a__2"]) == ["three"]
