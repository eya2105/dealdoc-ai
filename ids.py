"""Identifier helpers — pure functions, shared by ingestion and the API."""
import re


def slugify(value: str) -> str:
    """Normalize a user-supplied id to [a-z0-9_] (safe for Chroma ids, URLs and HTML)."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError(f"Invalid identifier: {value!r}")
    return slug


def make_chunk_id(deal_id: str, doc_id: str, index: int) -> str:
    """Globally unique chunk id. Slugs never contain '__', so the format is unambiguous."""
    return f"{deal_id}__{doc_id}__{index}"
