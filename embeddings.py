from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed a batch of chunks for storage."""
    return _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string for search."""
    return _model.encode([text], normalize_embeddings=True).tolist()[0]