from dotenv import load_dotenv
load_dotenv()
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# ── Embedding ──────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking ───────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ──────────────────────────────────────────────────
TOP_K = 5
CANDIDATES_PER_QUERY = 15  # fetch more, deduplicate later

# ── LLM ────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Storage ────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_deals"
COLLECTION_NAME = "deal_docs"
UPLOAD_PATH = "./uploads"
INGEST_BATCH_SIZE = 100
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

# ── Shared instances ───────────────────────────────────────────
model = SentenceTransformer(EMBEDDING_MODEL)
db = PersistentClient(path=CHROMA_PATH)
llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)