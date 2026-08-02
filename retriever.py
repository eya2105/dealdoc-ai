from typing import Optional
from config import db, model, TOP_K, COLLECTION_NAME

MAX_TOP_K = 20


def retrieve(query: str, top_k: int = TOP_K,
             deal_id: Optional[str] = None) -> list[dict]:
    """Semantic search over deal document chunks."""
    top_k = max(1, min(top_k, MAX_TOP_K))

    collection = db.get_or_create_collection(COLLECTION_NAME)
    query_embedding = model.encode([query]).tolist()

    where_filter = {"deal_id": deal_id} if deal_id else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": 1 - results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]
