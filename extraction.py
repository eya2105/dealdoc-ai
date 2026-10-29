"""Structured extraction.

For every requested field: one targeted retrieval, then one small LLM call that sees
only that field's evidence. The LLM returns a value plus the labels (S1, S2...) of the
chunks that support it; labels are mapped back to real chunk ids so each value can be
traced to its source. Small isolated calls keep the model focused and its citations short
(a single prompt with ~20 excerpts for 8 fields gave nulls and wrong labels).
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from config import EXTRACT_CHUNKS_PER_FIELD, EXTRACT_MAX_WORKERS, EXTRACT_POOL, MAX_EXTRACT_FIELDS
from ports import LLM, Chunk

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """The LLM did not return a usable extraction."""


EXTRACT_FIELD_SYSTEM_PROMPT = """You extract ONE field from M&A document excerpts.
You receive the field name, what to look for, and numbered excerpts (S1, S2, ...).

Return ONLY a JSON object: {"value": <the value, a list for multi-valued fields, or null>, "evidence": [<labels of the excerpts that state it>]}

RULES:
- Use ONLY the excerpts. Never use outside knowledge.
- Copy numbers, dates and names exactly as written. Do not estimate or compute.
- Documents use defined terms such as "Parent", "the Company" or "Sub". Report the real name given where the term is defined, never the defined term itself, but only if that definition appears in the excerpts.
- If the excerpts do not state the field, answer {"value": null, "evidence": []}.
- Cite only labels that appear in the excerpts, e.g. ["S2"]."""

FIELD_QUERY_SYSTEM_PROMPT = """You help search M&A legal documents (merger agreements, proxy statements).
For each field name, write a short search query (6 to 14 words) that reads like the sentence in the document which STATES that fact.
Mimic the document's own wording, not the topic name. Use generic legal wording and no company names.
Do not pick a different concept that merely shares a word with the field.

Examples (other fields):
- governing_law -> this agreement shall be governed by and construed in accordance with the laws of the State of
- closing_conditions -> obligations of each party to consummate the merger are subject to the satisfaction of the following conditions
- shares_outstanding -> shares of common stock issued and outstanding as of the record date

Return ONLY a JSON object: {"queries": {"<field name>": "<search query>"}}"""

# The model sometimes writes "None" or "N/A" as a string instead of JSON null.
NULL_STRINGS = {"none", "null", "n/a", "na", "unknown", "not stated", "not specified", "not available"}

# Search wording for the default fields: generic M&A language, no deal-specific names,
# and no defined terms like "Parent" (they match every clause that mentions Parent).
FIELD_QUERIES = {
    "deal_value": "total merger consideration per share paid in cash, purchase price",
    "buyer": "agreement and plan of merger by and among the acquiring corporation, merger subsidiary and the company",
    "target": "the Company being acquired, Merger Sub merges with and into the Company",
    "announcement_date": "agreement and plan of merger dated as of, announcement of the transaction",
    "deal_type": "structure of the transaction: merger, tender offer, all-cash or stock consideration",
    "industry": "description of the Company's business, products and industry",
    "advisors": "financial advisor rendered opinion to the board of directors, legal counsel engaged",
    "key_conditions": "conditions to completion of the merger: stockholder approval, regulatory approvals",
    "timeline": "expected timing of closing, special meeting date, record date, outside date",
}


# ── Pure helpers (no I/O, unit-tested directly) ─────────────────────────

def clean_fields(fields: list[str], max_fields: int = MAX_EXTRACT_FIELDS) -> list[str]:
    """Strip, drop blanks and duplicates (order kept); enforce the size limit."""
    cleaned = list(dict.fromkeys(f.strip() for f in fields if f and f.strip()))
    if not cleaned:
        raise ValueError("At least one field is required")
    if len(cleaned) > max_fields:
        raise ValueError(f"Too many fields (max {max_fields})")
    return cleaned


def build_field_context(field: str, query: str, chunks: list[Chunk]) -> tuple[str, dict[str, Chunk]]:
    """The prompt for one field: its name, what to look for, and its labelled excerpts."""
    chunk_of = {f"S{i}": chunk for i, chunk in enumerate(chunks, start=1)}
    excerpts = "\n\n".join(f"[{label}] {c['text']}" for label, c in chunk_of.items()) or "(no excerpt found)"
    return f"FIELD: {field}\nWHAT TO LOOK FOR: {query}\n\nEXCERPTS:\n{excerpts}", chunk_of


def parse_field_result(entry, chunk_of: dict[str, Chunk]) -> dict:
    """Validate one LLM answer: normalise empty values, keep only labels that really exist."""
    if isinstance(entry, dict) and "value" in entry:
        value, evidence = entry["value"], entry.get("evidence") or []
    else:  # the model returned a bare value instead of {"value", "evidence"}
        value, evidence = entry, []
    if value in ("", [], {}) or (isinstance(value, str) and value.strip().lower() in NULL_STRINGS):
        value = None
    if isinstance(evidence, str):
        evidence = [evidence]

    sources, seen = [], set()
    for label in evidence if value is not None else []:
        chunk = chunk_of.get(str(label).strip().strip("[]"))  # hallucinated labels are dropped
        if chunk and chunk["chunk_id"] not in seen:
            seen.add(chunk["chunk_id"])
            sources.append({"chunk_id": chunk["chunk_id"], "doc_id": chunk["metadata"]["doc_id"]})
    return {"value": value, "sources": sources}


@dataclass
class ExtractionRun:
    """Every intermediate of one extraction, so a wrong value can be traced to its stage."""
    fields: list[str]
    queries: dict[str, str]           # field -> search query used
    evidence: dict[str, list[Chunk]]  # field -> retrieved chunks
    contexts: dict[str, str]          # field -> prompt sent to the LLM
    raw: dict[str, dict]              # field -> the LLM's JSON, untouched
    results: dict[str, dict]          # field -> validated {value, sources}


# ── The use case ────────────────────────────────────────────────────────

class Extractor:
    def __init__(self, retriever, llm: LLM, *,
                 chunks_per_field: int = EXTRACT_CHUNKS_PER_FIELD,
                 pool: int = EXTRACT_POOL,
                 max_workers: int = EXTRACT_MAX_WORKERS,
                 max_fields: int = MAX_EXTRACT_FIELDS):
        self._retriever, self._llm = retriever, llm
        self._chunks_per_field, self._pool = chunks_per_field, pool
        self._max_workers, self._max_fields = max_workers, max_fields
        self._written_queries: dict[str, str] = {}  # LLM-written queries, kept for this Extractor's life

    def resolve_queries(self, fields: list[str]) -> dict[str, str]:
        """Known fields use FIELD_QUERIES; free-form ones get their own words + an LLM-written expansion.

        Keeping the user's words matters: a field name often IS the document's defined term
        ("company termination fee"), and the LLM's rewording can lose that exact match.
        The expansion adds the document's vocabulary on top (query + generated text).
        """
        unknown = [f for f in fields if f not in FIELD_QUERIES and f not in self._written_queries]
        if unknown:
            written = self._write_queries(unknown) or {}
            for field in unknown:
                query = written.get(field)
                if isinstance(query, str) and query.strip():
                    self._written_queries[field] = query.strip()

        queries = {}
        for field in fields:
            if field in FIELD_QUERIES:
                queries[field] = FIELD_QUERIES[field]
                continue
            own_words = field.replace("_", " ")
            expansion = self._written_queries.get(field)
            queries[field] = f"{own_words}. {expansion}" if expansion else own_words
        return queries

    def _write_queries(self, fields: list[str]) -> Optional[dict]:
        """One LLM call that turns free-form field names into search queries; None on failure."""
        data = self._llm.complete_json(FIELD_QUERY_SYSTEM_PROMPT, "Fields: " + ", ".join(fields), temperature=0.0)
        if not data or not isinstance(data.get("queries"), dict):
            return None
        return data["queries"]

    def retrieve_evidence(self, deal_id: str, fields: list[str], per_field: Optional[int] = None
                          ) -> tuple[list[str], dict[str, str], dict[str, list[Chunk]]]:
        """Everything before the LLM: clean the fields, choose a query per field, retrieve its chunks.

        One targeted retrieval per field, no query expansion (that would be one more LLM call each).
        """
        fields = clean_fields(fields, self._max_fields)
        queries = self.resolve_queries(fields)
        top_k = per_field or self._chunks_per_field
        evidence = {
            field: self._retriever.retrieve(query, top_k=top_k, deal_id=deal_id, expand=False, pool=self._pool)
            for field, query in queries.items()
        }
        return fields, queries, evidence

    def _ask(self, context: str) -> Optional[dict]:
        return self._llm.complete_json(EXTRACT_FIELD_SYSTEM_PROMPT, context, temperature=0.0)

    def run(self, deal_id: str, fields: list[str]) -> ExtractionRun:
        """retrieve per field -> one small LLM call per field (parallel) -> validate."""
        fields, queries, evidence = self.retrieve_evidence(deal_id, fields)

        contexts, chunk_ofs = {}, {}
        for field in fields:
            contexts[field], chunk_ofs[field] = build_field_context(field, queries[field], evidence[field])

        # no evidence -> no point asking the LLM
        to_ask = [f for f in fields if evidence[f]]
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            answers = list(pool.map(self._ask, [contexts[f] for f in to_ask]))
        raw = {f: {"value": None, "evidence": []} for f in fields}
        raw.update(dict(zip(to_ask, answers)))

        failed = [f for f in to_ask if raw[f] is None]
        if failed:
            raise ExtractionError(f"The LLM returned no valid JSON for: {', '.join(failed)}")

        results = {f: parse_field_result(raw[f], chunk_ofs[f]) for f in fields}
        return ExtractionRun(fields, queries, evidence, contexts, raw, results)

    def extract(self, deal_id: str, fields: list[str]) -> dict[str, dict]:
        """Extract `fields` from every document of a deal. Returns {field: {value, sources}}."""
        return self.run(deal_id, fields).results
