"""Hand-written fakes for the ports: no monkeypatching, no model, no API key."""
import hashlib
from types import SimpleNamespace


def make_chunk(chunk_id="c1", text="text", doc_id="doc", deal_id="deal", score=0.5):
    return {"chunk_id": chunk_id, "text": text, "score": score,
            "metadata": {"doc_id": doc_id, "deal_id": deal_id}}


def write_html(path, body):
    path.write_text(f"<html><body><p>{body}</p></body></html>", encoding="utf-8")
    return str(path)


class FakeEmbedder:
    """Deterministic 16-d normalized vectors: same text -> same vector."""

    def embed(self, texts):
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            v = [b / 255 + 0.01 for b in digest[:16]]
            norm = sum(x * x for x in v) ** 0.5
            out.append([x / norm for x in v])
        return out


class FakeLLM:
    """`json_fn(system, user)` decides what complete_json answers (None = failure)."""

    def __init__(self, text="answer", json_fn=None):
        self.text, self.json_fn = text, json_fn
        self.text_calls, self.json_calls = [], []

    def complete_text(self, system, user, temperature=0.1):
        self.text_calls.append((system, user))
        return self.text

    def complete_json(self, system, user, temperature=0.0):
        self.json_calls.append((system, user))
        return self.json_fn(system, user) if self.json_fn else None


class FakeReranker:
    """Scores 1.0 for chunks that mention 'needle', else 0.0 (override with score_fn)."""

    def __init__(self, score_fn=None):
        self.score_fn = score_fn or (lambda query, chunk: 1.0 if "needle" in chunk["text"] else 0.0)
        self.pool_sizes = []

    def rerank(self, query, candidates, top_k):
        self.pool_sizes.append(len(candidates))
        scored = [{**c, "rerank_score": float(self.score_fn(query, c))} for c in candidates]
        return sorted(scored, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


class FakeRetriever:
    """Records every call; `chunks_fn(query)` decides what comes back."""

    def __init__(self, chunks_fn=None, variants=None):
        self.chunks_fn = chunks_fn or (lambda query: [])
        self.variants = ["variant"] if variants is None else variants
        self.calls = []

    def retrieve(self, query, top_k=5, deal_id=None, doc_id=None, expand=None, rerank=None, pool=None):
        self.calls.append({"query": query, "top_k": top_k, "deal_id": deal_id, "doc_id": doc_id,
                           "expand": expand, "rerank": rerank, "pool": pool})
        return self.chunks_fn(query)

    def expand_query(self, question):
        return self.variants


def fake_client(content=None, error=None):
    """Stands in for the OpenAI client inside GroqLLM."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client
