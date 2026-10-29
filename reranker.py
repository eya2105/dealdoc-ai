"""Cross-encoder reranking (implements the Reranker port).

The embedding model scores the question and each chunk independently (fast, approximate).
A cross-encoder reads the question and a chunk together and scores that pair
(slower, more precise), so it is used only on the few dozen candidates the vector
search already found.
"""
from config import RERANK_MODEL_NAME


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL_NAME, model=None):
        self._model_name = model_name
        self._model = model  # injectable for tests

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # heavy import, keep it lazy

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Return the `top_k` candidates sorted by cross-encoder score (adds `rerank_score`)."""
        if not candidates:
            return []
        scores = self._get_model().predict(
            [(query, c["text"]) for c in candidates],
            show_progress_bar=False,
        )
        scored = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
        return sorted(scored, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
