from dotenv import load_dotenv
load_dotenv()

import os

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# Embedding 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval 
TOP_K = 5

# LLM 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1

# Storage 
CHROMA_PATH = "./chroma_deals"
COLLECTION_NAME = "deal_docs"
UPLOAD_PATH = "./uploads"
INGEST_BATCH_SIZE = 100

# Shared instances (initialized once, imported everywhere)
model = SentenceTransformer(EMBEDDING_MODEL)
db = PersistentClient(path=CHROMA_PATH)
