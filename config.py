"""Application configuration — constants only, no side effects beyond .env loading."""
import os

from dotenv import load_dotenv

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Embedding
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K = 5
MAX_TOP_K = 20
CANDIDATES_PER_QUERY = 30  # vector candidates per query for /query (a pool of 100 gave MRR 0.82 -> 0.80)

# Structured extraction
EXTRACT_CHUNKS_PER_FIELD = 4  # evidence chunks the LLM sees per field (3 missed one answer that sits at rank 4)
EXTRACT_POOL = 100  # candidates the reranker reads per field: hard-to-find facts need a deeper pool than Q&A
MAX_EXTRACT_FIELDS = 20
EXTRACT_MAX_WORKERS = 2  # parallel LLM calls (one per field); more does not help under a tokens-per-minute cap

# Reranking (cross-encoder re-scores the top RERANK_POOL candidates)
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 30
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true" 
QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() == "true"  

# LLM (the key is checked lazily in llm.py, so tools like tests and `eval.py --check-golden` can run without it)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")  # Groq retired llama-3.3-70b-versatile on 2026-08-16
LLM_TEMPERATURE = 0.1
LLM_MAX_RETRIES = 6  # on 429 (rate limit) the SDK waits Retry-After / backs off, then retries
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Storage
CHROMA_PATH = "./chroma_deals"
COLLECTION_NAME = "deal_docs"
COLLECTION_DISTANCE_SPACE = "cosine"  # Chroma defaults to l2
UPLOAD_PATH = "./uploads"
INGEST_BATCH_SIZE = 100
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".html", ".htm"}

# Documents loaded by `python ingest.py`
# deal_id = the transaction, doc_id = one document inside it
PRELOAD_DOCUMENTS = [
    {
        "file_path": "data/merger_agreement.html",
        "deal_id": "msft_activision",
        "doc_id": "merger_agreement",
        "doc_type": "merger_agreement",
    },
    {
        "file_path": "data/proxy_statement.html",
        "deal_id": "msft_activision",
        "doc_id": "proxy_statement",
        "doc_type": "proxy_statement",
    },
]

# Evaluation
GOLDEN_PATH = "eval/golden.json"
GOLDEN_EXTRACT_PATH = "eval/golden_extract.json"
EVAL_RESULTS_DIR = "eval/results"
