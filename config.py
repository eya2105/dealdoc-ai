"""Application configuration."""
from dotenv import load_dotenv
load_dotenv()

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Embedding 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 5
MAX_TOP_K = 20
CANDIDATES_PER_QUERY = 15  

# LLM 
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file before starting the app."
    )

# Storage
CHROMA_PATH = "./chroma_deals"
COLLECTION_NAME = "deal_docs"
COLLECTION_DISTANCE_SPACE = "cosine"  # Chroma defaults to l2 
UPLOAD_PATH = "./uploads"
INGEST_BATCH_SIZE = 100  # Chroma 0.4.x caps add() batches around ~166 items
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB