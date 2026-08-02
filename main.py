import logging
from pathlib import Path
from typing import Optional
 
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
import config
from ingest import ingest
from retriever import retrieve
from generator import generate_answer, extract_structured
from store import get_collection
 
logging.basicConfig(level=logging.INFO)
 
app = FastAPI(
    title="DealDoc AI",
    description="M&A Document Intelligence API — RAG-powered deal analysis",
    version="0.1.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# Request / Response Models
 
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    deal_id: Optional[str] = None
 
 
class ExtractRequest(BaseModel):
    deal_id: str
    fields: list[str] = [
        "deal_value", "buyer", "target",
        "announcement_date", "deal_type", "industry",
        "advisors", "key_conditions", "timeline",
    ]
 
 
class IngestResponse(BaseModel):
    deal_id: str
    total_chunks: int
    document_chars: int
 
 
class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
 
 
class ExtractResponse(BaseModel):
    deal_id: str
    extracted_fields: dict
 
 
# Endpoints
 
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    deal_id: Optional[str] = Query(None),
    deal_type: Optional[str] = Query(None),
):
    content = await file.read()
    if len(content) > config.MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Max {config.MAX_UPLOAD_SIZE // (1024*1024)}MB")
 
    safe_filename = Path(file.filename).name
    save_path = Path(config.UPLOAD_PATH) / safe_filename
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, "wb") as f:
        f.write(content)
 
    metadata = {
        "deal_id": deal_id or save_path.stem,
        "source_file": safe_filename,
        "deal_type": deal_type or "unknown",
    }
 
    try:
        result = ingest(str(save_path), metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
 
    return IngestResponse(**result)
 
 
@app.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    results = retrieve(req.question, top_k=req.top_k, deal_id=req.deal_id)
 
    if not results:
        return QueryResponse(answer="No relevant chunks found.", sources=[])
 
    contexts = [r["text"] for r in results]
    answer = generate_answer(req.question, contexts)
 
    sources = [
        {
            "chunk_id": r["chunk_id"],
            "deal_id": r["metadata"]["deal_id"],
            "score": round(r["score"], 4),
            "text_preview": r["text"][:150] + "...",
        }
        for r in results
    ]
    return QueryResponse(answer=answer, sources=sources)
 
 
@app.post("/extract", response_model=ExtractResponse)
async def extract_deal_terms(req: ExtractRequest):
    collection = get_collection()
    existing = collection.get(where={"deal_id": req.deal_id}, include=[])
    if not existing["ids"]:
        raise HTTPException(404, f"Deal '{req.deal_id}' not found")
 
    results = retrieve(f"Extract: {', '.join(req.fields)}", top_k=15, deal_id=req.deal_id)
    relevant_chunks = [r["text"] for r in results]
 
    extracted = extract_structured(relevant_chunks, req.fields)
    return ExtractResponse(deal_id=req.deal_id, extracted_fields=extracted)
 
 
@app.get("/documents")
async def list_documents():
    collection = get_collection()
    all_meta = collection.get(include=["metadatas"])["metadatas"]
 
    seen = set()
    docs = []
    for m in all_meta:
        did = m["deal_id"]
        if did not in seen:
            seen.add(did)
            docs.append({
                "deal_id": did,
                "source_file": m.get("source_file", "unknown"),
                "deal_type": m.get("deal_type", "unknown"),
            })
    return {"documents": docs, "total": len(docs)}
 
 
@app.get("/health")
async def health():
    try:
        get_collection()
        return {"status": "ok", "service": "dealdoc-ai"}
    except Exception:
        return {"status": "error", "service": "dealdoc-ai"}