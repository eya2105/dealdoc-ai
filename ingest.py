import re
from pathlib import Path
from bs4 import BeautifulSoup
from config import (
    db, model, CHUNK_SIZE, CHUNK_OVERLAP,
    COLLECTION_NAME, INGEST_BATCH_SIZE,
)


def parse_document(file_path: str) -> str:
    """Extract clean text from an HTML SEC filing."""
    with open(file_path, "r", encoding="windows-1252", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    raw = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", raw).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start: start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest(file_path: str, metadata: dict) -> dict:
    """Full ingestion pipeline: parse → chunk → embed → store."""
    text = parse_document(file_path)
    chunks = chunk_text(text)

    embeddings = model.encode(
        chunks,
        show_progress_bar=False,
        batch_size=32,
    ).tolist()

    collection = db.get_or_create_collection(COLLECTION_NAME)

    doc_id = metadata.get("deal_id", Path(file_path).stem)
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

    # ChromaDB 0.4.x limits each add() to 166 items — batch accordingly
    for i in range(0, len(chunks), INGEST_BATCH_SIZE):
        end = i + INGEST_BATCH_SIZE
        collection.add(
            ids=ids[i:end],
            documents=chunks[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    return {
        "deal_id": doc_id,
        "total_chunks": len(chunks),
        "document_chars": len(text),
    }



#preload 
if __name__ == "__main__":
    docs = [
        {
            "path": "data/Agreement and Plan of Merger.html",
            "deal_id": "msft_activision_merger",
            "deal_type": "merger_agreement",
        },
        {
            "path": "data/DEF M 14A — Special Meeting Proxy Statement.html",
            "deal_id": "msft_activision_proxy",
            "deal_type": "proxy_statement",
        },
    ]
    for doc in docs:
        result = ingest(doc["path"], {
            "deal_id": doc["deal_id"],
            "deal_type": doc["deal_type"],
            "source_file": Path(doc["path"]).name,
        })
        print(f"Ingested {result['total_chunks']} chunks for {result['deal_id']}")
