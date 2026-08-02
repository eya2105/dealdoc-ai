import json
from typing import Optional
from config import (
    db, model, llm_client, TOP_K,
    COLLECTION_NAME, CANDIDATES_PER_QUERY, LLM_MODEL,
)

MAX_TOP_K = 20


def expand_query(question: str) -> list[str]:
    """Generate alternative phrasings to improve retrieval recall.

    A single question often misses relevant chunks because the
    embedding model matches on surface similarity. By generating
    2-3 rephrased variants, we cast a wider net.
    """
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": (
                    "You rewrite user questions for a legal document search engine. "
                    "Generate 2 alternative phrasings that use different terms "
                    "but preserve the same intent. "
                    'Return ONLY a JSON object: {"queries": ["...", "..."]}'
                )},
                {"role": "user", "content": question},
            ],
        )
    except Exception:
        return []

    try:
        data = json.loads(response.choices[0].message.content)
        return data.get("queries", [])
    except (json.JSONDecodeError, AttributeError):
        return []


def retrieve(query: str, top_k: int = TOP_K,
             deal_id: Optional[str] = None) -> list[dict]:
    """Semantic search with multi-query expansion.

    1. Generate 2-3 query variants via LLM
    2. Embed each variant and retrieve candidates from ChromaDB
    3. Deduplicate by chunk_id, keep the best score per chunk
    4. Sort by score descending, return top_k
    """
    top_k = max(1, min(top_k, MAX_TOP_K))
    collection = db.get_or_create_collection(COLLECTION_NAME)
    where_filter = {"deal_id": deal_id} if deal_id else None

    # Build list of queries: original + expanded variants
    variants = expand_query(query)
    queries = [query] + variants

    # Retrieve candidates for each query, deduplicate by chunk_id
    seen: dict[str, dict] = {}
    for q in queries:
        embedding = model.encode([q]).tolist()
        results = collection.query(
            query_embeddings=embedding,
            n_results=CANDIDATES_PER_QUERY,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            distance = results["distances"][0][i]

            # Keep the chunk with the best (lowest) distance
            if chunk_id not in seen or distance < seen[chunk_id]["distance"]:
                seen[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i],
                    "distance": distance,
                    "score": 1 - distance,
                    "metadata": results["metadatas"][0][i],
                }

    # Sort by similarity score descending, take top_k
    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]