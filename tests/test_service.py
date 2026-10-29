import pytest

from service import DealNotFoundError
from tests.fakes import write_html


def test_ask_returns_the_answer_with_numbered_sources(container, tmp_path):
    container.ingestor.ingest(write_html(tmp_path / "a.html", "price is $95.00 per share"), deal_id="d", doc_id="a")
    out = container.service.ask("price per share?", top_k=3, deal_id="d")
    assert out["answer"] == "answer [Source 1]"
    assert [s["rank"] for s in out["sources"]] == list(range(1, len(out["sources"]) + 1))
    assert out["sources"][0]["doc_id"] == "a"
    assert out["sources"][0]["text_preview"].endswith("...")
    assert out["sources"][0]["text"] == "price is $95.00 per share"   # the full passage, not only a preview
    assert out["sources"][0]["source_file"] == "a.html"
    _, user_prompt = container.llm.text_calls[0]
    assert "[Source 1]: price is $95.00 per share" in user_prompt  # the generator received the chunk text


def test_ask_without_chunks_does_not_call_the_llm(container):
    out = container.service.ask("anything", deal_id="no-such-deal")
    assert out == {"answer": "No relevant chunks found.", "sources": []}
    assert container.llm.text_calls == []


def test_extract_unknown_deal_raises(container):
    with pytest.raises(DealNotFoundError):
        container.service.extract("ghost", ["buyer"])


def test_extract_known_deal_uses_the_extractor(container, tmp_path):
    container.ingestor.ingest(write_html(tmp_path / "a.html", "Microsoft Corporation acquires the Company"),
                              deal_id="d", doc_id="a")
    out = container.service.extract("d", ["buyer"])
    assert out["buyer"]["value"] == "Microsoft"


def test_documents_and_health(container, tmp_path):
    container.ingestor.ingest(write_html(tmp_path / "a.html", "some text"), deal_id="d", doc_id="a")
    assert [d["doc_id"] for d in container.service.documents()] == ["a"]
    container.service.check_health()  # does not raise