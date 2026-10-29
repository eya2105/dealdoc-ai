# DealDoc AI

RAG-powered document intelligence for M&A filings: ingest SEC HTML filings, ask questions in natural language, and extract structured deal terms. Every answer and every extracted value points back to the chunks it came from.

Built and evaluated on two public documents of one deal (Microsoft / Activision Blizzard: the merger agreement and the proxy statement).

## How it works

1. **Ingest**: parse HTML, chunk (1000 chars, 200 overlap, word boundaries), embed (`all-MiniLM-L6-v2`), store in ChromaDB (cosine). Re-ingesting a document replaces it.
2. **Retrieve**: vector search over 30 candidates, then a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) keeps the best chunks. LLM query expansion exists but is off by default (see results).
3. **Answer**: an LLM (Groq, default `openai/gpt-oss-120b`) answers only from the retrieved chunks and cites `[Source N]`.
4. **Extract**: per requested field, one targeted retrieval and one small LLM call; each value comes back with the chunk ids that support it.

Data model: a **deal** (`deal_id`) contains **documents** (`doc_id`). Queries can target everything, one deal, or one document.

## Results (measured with the scripts in this repo)

### Retrieval: `python eval.py` (26 hand-written questions, k=5)

| Setup | recall@5 | MRR |
|---|---|---|
| vector search only | 0.65 | 0.39 |
| + cross-encoder reranker | 0.88 | 0.77 |
| + reranker, after correcting two golden entries | 0.96 | 0.82 |
| reranker + LLM query expansion (one run) | 0.81 | 0.69 |
| reranker, candidate pool of 100 instead of 30 | 0.96 | 0.80 |

- The reranker is the one change that clearly helped (deterministic runs, +6 questions).
- LLM query expansion did not help here (slightly worse, one nondeterministic run), so it is opt-in.
- A bigger candidate pool (100) did not raise recall and lowered MRR slightly, so Q&A keeps a pool of 30; only extraction uses 100 (see below).
- The 0.88 to 0.96 step comes from fixing two golden-set entries whose expected strings were wrong, not from a code change. Those edits were made after seeing results.

### Extraction: `python eval_extract.py` (8 fields, one run)

7 of 8 fields correct and grounded (the cited chunks really contain the value). Progression while debugging: 3/8, then 5/8 and 4/8 (the LLM varies between runs), 6/8 after per-field retrieval, per-field LLM calls and better field queries, and 7/8 after measuring the rank of each answer chunk (`--retrieval-only --depth 20`): one date sat at rank 4 and outside the 30-candidate pool, so extraction now reads 4 chunks per field from a 100-candidate pool.

Correctness is "the value contains the expected string", not exact match: for example one date came back as `11:59 p.m., Pacific time, on January 18, 2023`. A run takes about 80 seconds on the free Groq tier because of the tokens-per-minute cap.

## Known limitations

- **Small evaluation.** 26 questions and 8 extraction fields on a single deal. It shows where the method works (numbers, defined dates, named entities) and where it fails; it does not prove generalization.
- **One extraction field still fails**: the buyer's financial advisor. The fact is mentioned only in passing in the negotiation narrative; it is not in the top 20 chunks even with a 100-candidate pool, and the retrieved chunks are the generic "no financial advisor was retained" clause. This is a retrieval limit of similarity search, not an LLM error (the LLM correctly answers null).
- **Fixed-size chunking** ignores sections and tables. Pure vector retrieval is weak on exact terms (no BM25 / hybrid search yet).
- **Rate limits.** On the free Groq tier (8000 tokens per minute for the default model) one extraction takes about 40 seconds. The client retries on 429.
- **Confidentiality.** Documents are sent to an external LLM API, acceptable for public SEC filings; confidential deals would need a self-hosted model.
- **Not production-ready:** no authentication, CORS open to all origins.

## Design

The code follows ports and adapters: the logic depends on small interfaces, and the libraries sit behind them.

```
main.py (routes) -> service.py (RagService: the use cases)
                      |-> Ingestor, Retriever, AnswerGenerator, Extractor   (the logic)
                              |-> ports.py: Embedder, Reranker, LLM, VectorIndex, DocumentCatalog
                                      ^ implemented by: SentenceTransformerEmbedder, CrossEncoderReranker,
                                        GroqLLM, ChromaStore
container.py builds the real objects (the only place that knows which ones)
```

- **Dependency inversion and injection.** `Retriever`, `Extractor`, `Ingestor`... receive their collaborators in the constructor and only know the `Protocol`s in `ports.py`. Replacing Groq by a local model, or Chroma by another database, means writing one adapter and changing `container.py`.
- **Single responsibility.** Parsing, chunking, embedding, storage, retrieval, generation, extraction and HTTP are separate modules. Routes only parse requests and map errors to status codes.
- **Open for extension.** `Ingestor` takes the parser and the chunking strategy as arguments, so a new format or a section-aware chunker is a new function, not an edit.
- **Interface segregation.** Search and write (`VectorIndex`) are separate from catalog queries (`DocumentCatalog`).
- **DRY.** One metadata filter builder (`build_where`), one way to call the LLM (`GroqLLM._create`), one retrieval path shared by Q&A, extraction and the evaluation scripts.
- **Testability.** Tests use hand-written fakes (`tests/fakes.py`) through the real wiring (`build_container(...)`): no monkeypatching, no model, no API key, real Chroma on a temp directory.

## Project structure

```
main.py             FastAPI routes (thin)
service.py          RagService: ingest / ask / extract / documents
container.py        composition root: builds and wires everything
ports.py            the interfaces (typing.Protocol)
config.py           constants and feature flags
parsing.py          HTML -> text
chunking.py         pure chunking logic
ids.py              slugify / chunk ids
embeddings.py       SentenceTransformerEmbedder   (Embedder)
reranker.py         CrossEncoderReranker          (Reranker)
llm.py              GroqLLM                       (LLM)
store.py            ChromaStore                   (VectorIndex, DocumentCatalog)
ingest.py           Ingestor: parse -> chunk -> embed -> store
retriever.py        Retriever: vector search [+ expansion] [+ rerank]
generator.py        AnswerGenerator: grounded, cited answers
extraction.py       Extractor: per-field retrieval + LLM call + validation
eval.py             retrieval evaluation (recall@k, MRR)
eval_extract.py     extraction evaluation (correct / grounded / failing stage)
eval/               golden question sets
tests/              pytest suite and fakes
frontend.html       browser UI
```

## Run locally (Python 3.11)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env            # then put your GROQ_API_KEY in .env
python ingest.py                  # indexes the files listed in config.PRELOAD_DOCUMENTS
uvicorn main:app --reload
```

Open `frontend.html`, or Swagger at <http://localhost:8000/docs>.

Settings (environment variables / `.env`; pools and chunk counts are in `config.py`): `LLM_MODEL`, `RERANK_ENABLED` (default `true`), `QUERY_EXPANSION_ENABLED` (default `false`).

## Tests and evaluation

```bash
pytest -q                                   # unit + integration tests (fakes through the real wiring)
python eval.py --check-golden               # is every expected answer inside a single chunk?
python eval.py --rerank --tag v3            # retrieval metrics; add --show-misses, --expand
python eval.py --compare v2_rerank v3       # side by side
python eval_extract.py --retrieval-only     # extraction, retrieval stage only (no LLM quota)
python eval_extract.py --runs 2 --diagnose  # full extraction; says which stage failed per field
```

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/query` | Q&A with numbered source citations (`deal_id`, `doc_id` filters) |
| POST | `/extract` | Per field: `value` plus the chunks it was read from |
| POST | `/ingest?deal_id=&doc_id=&doc_type=` | Upload and index an HTML filing |
| GET | `/documents` | List indexed documents with chunk counts |
| GET | `/health` | Service status |
