"""Embedding adapter (implements the Embedder port). The model is loaded on first use."""
from config import EMBEDDING_MODEL_NAME


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, batch_size: int = 32):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Same model for chunks (storage) and queries (search)."""
        return self._get_model().encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

    @property
    def max_seq_length(self) -> int:
        """Max tokens the model encodes; anything longer is silently truncated."""
        return self._get_model().max_seq_length
