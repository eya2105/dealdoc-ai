# DealDoc AI

RAG-powered document intelligence system : ingest legal filings, ask questions in natural language, and extract structured data, with every answer grounded in source documents.

## What It Does

1. **Ingest**  Upload SEC HTML filings. Text is parsed, split into overlapping chunks, and stored as vectors in ChromaDB.
2. **Query**  Ask a question in plain English. The system retrieves the most relevant chunks, sends them to an LLM, and returns a cited answer.
3. **Extract**  Pull structured fields (deal value, parties, dates) from any indexed document as JSON.

- **Chunking:** 1000 chars with 200 char overlap, word-boundary snapping
- **Embedding:** `all-MiniLM-L6-v2` (384-dim), cosine similarity
- **Retrieval:** Multi-query expansion : the LLM rewrites the question 2-3 ways to improve recall, then results are deduplicated by best score
- **Generation:** `llama-3.3-70b-versatile` via Groq, temperature 0.1 for Q&A, 0.0 for JSON extraction

## Project Structure

```
dealdoc-ai/
├── main.py            # FastAPI endpoints 
├── config.py          # Constants and settings
├── embeddings.py      # Embedding model wrapper
├── llm.py             # LLM client (Groq)
├── store.py           # ChromaDB access
├── ingest.py          # Parse → chunk → embed → store
├── retriever.py       # Multi-query semantic search
├── generator.py       # Q&A generation + structured extraction
├── frontend.html      # Browser UI
├── data/              # Source documents
└── requirements.txt
```

## Tech Stack

FastAPI · ChromaDB · Sentence-Transformers · Groq (llama-3.3-70b) · BeautifulSoup

## Setup

```bash
python -m venv myenv && myenv\Scripts\activate
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_..." > .env
python ingest.py
uvicorn main:app --reload
```

Open `frontend.html` in your browser, or use Swagger docs at `http://localhost:8000/docs`.

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/query` | Semantic Q&A with source citations |
| POST | `/extract` | Structured field extraction as JSON |
| POST | `/ingest` | Upload and index a new document |
| GET | `/documents` | List indexed documents |
| GET | `/health` | Service status |
