from typing import Optional

from config import (CANDIDATES_PER_QUERY, MAX_TOP_K, QUERY_EXPANSION_ENABLED,
                    RERANK_ENABLED, RERANK_POOL, TOP_K)
from ports import LLM, Chunk, Embedder, Reranker, VectorIndex

EXPAND_SYSTEM_PROMPT = (
    "You rewrite user questions for a legal document search engine. "
    "Generate 2 alternative phrasings that use different terms "
    "but preserve the same intent. "
    'Return ONLY a JSON object: {"queries": ["...", "..."]}'
)


class Retriever:
    """Semantic search: [query expansion] -> vector candidates -> [cross-encoder rerank].

    It depends only on ports (VectorIndex, Embedder, LLM, Reranker), never on a concrete library.
    """

    def __init__(self, index: VectorIndex, embedder: Embedder, llm: LLM, reranker: Reranker, *,
                 rerank_enabled: bool = RERANK_ENABLED,
                 expansion_enabled: bool = QUERY_EXPANSION_ENABLED,
                 candidates_per_query: int = CANDIDATES_PER_QUERY,
                 rerank_pool: int = RERANK_POOL,
                 max_top_k: int = MAX_TOP_K):
        self._index, self._embedder, self._llm, self._reranker = index, embedder, llm, reranker
        self._rerank_enabled = rerank_enabled
        self._expansion_enabled = expansion_enabled
        self._candidates_per_query = candidates_per_query
        self._rerank_pool = rerank_pool
        self._max_top_k = max_top_k

    def expand_query(self, question: str) -> list[str]:
        """Generate alternative phrasings to improve retrieval recall (one LLM call)."""
        data = self._llm.complete_json(EXPAND_SYSTEM_PROMPT, question, temperature=0.3)
        if not data:
            return []
        variants = data.get("queries", [])
        return [q for q in variants if isinstance(q, str) and q.strip()][:3]

    def _candidates(self, queries: list[str], deal_id: Optional[str], doc_id: Optional[str],
                    n_results: int) -> list[Chunk]:
        """One batched search; dedupe by chunk_id keeping the best score; sorted best-first."""
        per_query = self._index.search(self._embedder.embed(queries), n_results, deal_id, doc_id)
        best: dict[str, Chunk] = {}
        for chunks in per_query:
            for chunk in chunks:
                if chunk["chunk_id"] not in best or chunk["score"] > best[chunk["chunk_id"]]["score"]:
                    best[chunk["chunk_id"]] = chunk
        return sorted(best.values(), key=lambda c: c["score"], reverse=True)

    def retrieve(self, query: str,
                 top_k: int = TOP_K,
                 deal_id: Optional[str] = None,
                 doc_id: Optional[str] = None,
                 expand: Optional[bool] = None,
                 rerank: Optional[bool] = None,
                 pool: Optional[int] = None) -> list[Chunk]:
        """Best `top_k` chunks for `query`.

        `expand`: add LLM-written query variants (one extra LLM call); None = constructor default.
        `rerank`: re-score the best candidates with the cross-encoder; None = constructor default.
        `pool`:   how many candidates to fetch per query AND hand to the reranker
                  (default: candidates_per_query / rerank_pool). Extraction uses a deeper one.
        Results always carry `score` (cosine); reranked ones also carry `rerank_score`.
        """
        top_k = max(1, min(top_k, self._max_top_k))
        use_rerank = self._rerank_enabled if rerank is None else rerank
        use_expand = self._expansion_enabled if expand is None else expand

        queries = [query] + (self.expand_query(query) if use_expand else [])
        candidates = self._candidates(queries, deal_id, doc_id, pool or self._candidates_per_query)

        if use_rerank:
            # rerank against the ORIGINAL question, not the LLM variants
            return self._reranker.rerank(query, candidates[:pool or self._rerank_pool], top_k)
        return candidates[:top_k]
