"""Chroma adapter: implements VectorIndex and DocumentCatalog. The only module that knows Chroma."""
import threading
from typing import Optional

from chromadb import PersistentClient

from config import CHROMA_PATH, COLLECTION_NAME, COLLECTION_DISTANCE_SPACE, INGEST_BATCH_SIZE
from ports import Chunk


def build_where(deal_id: Optional[str] = None,
                doc_id: Optional[str] = None) -> Optional[dict]:
    """Build a Chroma metadata filter. Chroma needs $and for 2+ conditions."""
    conditions = [{k: v} for k, v in (("deal_id", deal_id), ("doc_id", doc_id)) if v]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class ChromaStore:
    def __init__(self, path: str = CHROMA_PATH, collection_name: str = COLLECTION_NAME,
                 space: str = COLLECTION_DISTANCE_SPACE, batch_size: int = INGEST_BATCH_SIZE):
        if space != "cosine":
            raise ValueError("Scores are computed as 1 - distance, which assumes cosine distance")
        self._path, self._name, self._space, self._batch_size = path, collection_name, space, batch_size
        self._collection = None
        self._lock = threading.Lock()

    @property
    def collection(self):
        """Opened on first use, so building the application is cheap."""
        if self._collection is None:
            with self._lock:
                if self._collection is None:
                    client = PersistentClient(path=self._path)
                    self._collection = client.get_or_create_collection(
                        self._name, metadata={"hnsw:space": self._space})
        return self._collection

    # ── VectorIndex ─────────────────────────────────────────────────────

    def replace_document(self, deal_id: str, doc_id: str, ids: list[str], texts: list[str],
                         embeddings: list[list[float]], metadatas: list[dict]) -> None:
        """Drop the old chunks of this document (a shorter new version leaves no stale ones), then add."""
        self.collection.delete(where=build_where(deal_id, doc_id))
        for i in range(0, len(ids), self._batch_size):
            end = i + self._batch_size
            self.collection.upsert(ids=ids[i:end], documents=texts[i:end],
                                   embeddings=embeddings[i:end], metadatas=metadatas[i:end])

    def search(self, query_embeddings: list[list[float]], n_results: int,
               deal_id: Optional[str] = None, doc_id: Optional[str] = None) -> list[list[Chunk]]:
        res = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=build_where(deal_id, doc_id),
            include=["documents", "metadatas", "distances"],
        )
        return [
            [{"chunk_id": cid, "text": text, "score": 1 - dist, "metadata": meta}
             for cid, text, meta, dist in zip(ids, texts, metas, dists)]
            for ids, texts, metas, dists in zip(
                res["ids"], res["documents"], res["metadatas"], res["distances"])
        ]

    # ── DocumentCatalog ─────────────────────────────────────────────────

    def deal_exists(self, deal_id: str) -> bool:
        found = self.collection.get(where=build_where(deal_id), limit=1, include=[])
        return bool(found["ids"])

    def list_documents(self) -> list[dict]:
        """One entry per (deal_id, doc_id), with its chunk count."""
        metadatas = self.collection.get(include=["metadatas"])["metadatas"]
        docs: dict[tuple[str, str], dict] = {}
        for m in metadatas:
            doc = docs.setdefault((m["deal_id"], m["doc_id"]), {
                "deal_id": m["deal_id"],
                "doc_id": m["doc_id"],
                "doc_type": m.get("doc_type", "unknown"),
                "source_file": m.get("source_file", "unknown"),
                "chunks": 0,
            })
            doc["chunks"] += 1
        return sorted(docs.values(), key=lambda d: (d["deal_id"], d["doc_id"]))

    def get_texts(self, ids: list[str]) -> list[str]:
        return self.collection.get(ids=ids, include=["documents"])["documents"] if ids else []

    def ping(self) -> None:
        self.collection.count()
