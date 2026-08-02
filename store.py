"""Chroma access — the only module that knows about the vector DB.
Every other module asks this one for a collection handle instead of
calling PersistentClient / get_or_create_collection itself."""
from chromadb import PersistentClient
from config import CHROMA_PATH, COLLECTION_NAME, COLLECTION_DISTANCE_SPACE

_client = PersistentClient(path=CHROMA_PATH)


def get_collection():
    """Return the shared deal-docs collection, created with cosine distance."""
    return _client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": COLLECTION_DISTANCE_SPACE},
    )