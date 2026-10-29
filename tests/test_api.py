import pytest
from fastapi.testclient import TestClient

import config
from container import get_rag_service
from main import app

HTML = b"<html><body><p>" + b" ".join(b"merger%d" % i for i in range(400)) + b"</p></body></html>"


@pytest.fixture
def client(container, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_PATH", str(tmp_path / "uploads"))
    app.dependency_overrides[get_rag_service] = lambda: container.service
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, name="deal.html", **params):
    return client.post("/ingest", params=params, files={"file": (name, HTML, "text/html")})


def test_ingest_then_list_then_query(client):
    r = upload(client, deal_id="MSFT Deal", doc_id="Merger Agreement", doc_type="merger_agreement")
    assert r.status_code == 200
    body = r.json()
    assert (body["deal_id"], body["doc_id"]) == ("msft_deal", "merger_agreement")

    docs = client.get("/documents").json()
    assert docs["total"] == 1 and docs["documents"][0]["chunks"] == body["total_chunks"]

    q = client.post("/query", json={"question": "merger5", "deal_id": "msft_deal"}).json()
    assert q["answer"] == "answer [Source 1]"
    assert [s["rank"] for s in q["sources"]] == list(range(1, len(q["sources"]) + 1))
    assert q["sources"][0]["doc_id"] == "merger_agreement"
    first = q["sources"][0]
    assert first["source_file"] == "deal.html"
    assert first["text"].startswith(first["text_preview"].removesuffix("..."))  # the full passage, not just the preview


def test_reingest_same_document_does_not_duplicate(client):
    upload(client, deal_id="d", doc_id="a")
    upload(client, deal_id="d", doc_id="a")
    assert client.get("/documents").json()["total"] == 1


def test_rejects_unsupported_extension(client):
    assert upload(client, name="notes.pdf").status_code == 400


def test_rejects_invalid_id(client):
    assert upload(client, deal_id="!!!").status_code == 422


def test_path_traversal_filename_is_neutralized(client, tmp_path):
    r = upload(client, name="../../evil.html", deal_id="d")
    assert r.status_code == 200
    assert (tmp_path / "uploads" / "evil.html").exists()


def test_query_validates_top_k(client):
    assert client.post("/query", json={"question": "x", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"question": "x", "top_k": 999}).status_code == 422


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


# ── /extract ────────────────────────────────────────────────────────────

def test_extract_unknown_deal_returns_404(client):
    assert client.post("/extract", json={"deal_id": "ghost"}).status_code == 404


def test_extract_returns_value_and_sources(client):
    upload(client, deal_id="d", doc_id="a")
    r = client.post("/extract", json={"deal_id": "d", "fields": ["buyer"]})
    assert r.status_code == 200
    field = r.json()["extracted_fields"]["buyer"]
    assert field["value"] == "Microsoft"
    assert field["sources"] and field["sources"][0]["doc_id"] == "a"


def test_extract_llm_failure_is_502(client, container):
    upload(client, deal_id="d", doc_id="a")
    container.llm.json_fn = lambda s, u: None
    assert client.post("/extract", json={"deal_id": "d", "fields": ["buyer"]}).status_code == 502


def test_extract_with_no_fields_is_422(client):
    upload(client, deal_id="d", doc_id="a")
    assert client.post("/extract", json={"deal_id": "d", "fields": [" "]}).status_code == 422


def test_root_serves_the_web_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "DealDoc" in r.text and 'id="btnQuery"' in r.text