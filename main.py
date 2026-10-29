import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from container import get_rag_service
from extraction import ExtractionError
from service import DealNotFoundError, RagService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="DealDoc AI",
    description="M&A Document Intelligence API — RAG-powered deal analysis",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ───────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=config.TOP_K, ge=1, le=config.MAX_TOP_K)
    deal_id: Optional[str] = None  # restrict to one transaction
    doc_id: Optional[str] = None   # restrict to one document of that transaction


class ExtractRequest(BaseModel):
    deal_id: str
    fields: list[str] = [
        "deal_value", "buyer", "target",
        "announcement_date", "deal_type", "industry",
        "advisors", "key_conditions", "timeline",
    ]


class IngestResponse(BaseModel):
    deal_id: str
    doc_id: str
    total_chunks: int
    document_chars: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class SourceRef(BaseModel):
    chunk_id: str
    doc_id: str


class FieldResult(BaseModel):
    value: Any = None              # null when the documents do not state it
    sources: list[SourceRef] = []  # chunks the value was read from


class ExtractResponse(BaseModel):
    deal_id: str
    extracted_fields: dict[str, FieldResult]


FRONTEND = Path(__file__).parent / "frontend.html"


@app.get("/", include_in_schema=False)
def frontend():
    """The web page, served by the API itself: open http://127.0.0.1:8000/"""
    return FileResponse(FRONTEND)


# ── Routes: parse the request, call the service, map errors to status codes ──
# Plain `def` (not `async def`): the work inside is blocking (embedding, Chroma, LLM),
# so FastAPI runs each call in a worker thread instead of freezing the event loop.

@app.post("/ingest", response_model=IngestResponse)
def ingest_document(
    file: UploadFile = File(...),
    deal_id: Optional[str] = Query(None),
    doc_id: Optional[str] = Query(None),
    doc_type: Optional[str] = Query(None),
    service: RagService = Depends(get_rag_service),
):
    filename = Path(file.filename or "").name  # strips any path -> no traversal
    if Path(filename).suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}")

    content = file.file.read()
    if len(content) > config.MAX_UPLOAD_SIZE:
        raise HTTPException(413, f"File too large. Max {config.MAX_UPLOAD_SIZE // (1024 * 1024)}MB")

    save_path = Path(config.UPLOAD_PATH) / filename
    save_path.parent.mkdir(exist_ok=True, parents=True)
    save_path.write_bytes(content)

    try:
        result = service.ingest(str(save_path), deal_id=deal_id, doc_id=doc_id,
                                doc_type=doc_type or "unknown")
    except ValueError as e:  # bad id, empty document
        raise HTTPException(422, str(e))
    except Exception:
        logger.exception("Ingestion failed for %s", filename)
        raise HTTPException(500, "Ingestion failed — see server logs")
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest, service: RagService = Depends(get_rag_service)):
    return QueryResponse(**service.ask(req.question, top_k=req.top_k,
                                       deal_id=req.deal_id, doc_id=req.doc_id))


@app.post("/extract", response_model=ExtractResponse)
def extract_endpoint(req: ExtractRequest, service: RagService = Depends(get_rag_service)):
    try:
        fields = service.extract(req.deal_id, req.fields)
    except DealNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:  # no fields / too many fields
        raise HTTPException(422, str(e))
    except ExtractionError as e:
        logger.error("Extraction failed for deal %s: %s", req.deal_id, e)
        raise HTTPException(502, f"Extraction failed: {e}")
    return ExtractResponse(deal_id=req.deal_id, extracted_fields=fields)


@app.get("/documents")
def documents(service: RagService = Depends(get_rag_service)):
    docs = service.documents()
    return {"documents": docs, "total": len(docs)}


@app.get("/health")
def health(service: RagService = Depends(get_rag_service)):
    try:
        service.check_health()
        return {"status": "ok", "service": "dealdoc-ai"}
    except Exception:
        logger.exception("Health check failed")
        return {"status": "error", "service": "dealdoc-ai"}