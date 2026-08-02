from config import LLM_TEMPERATURE
from llm import call_llm_json, call_llm_text

SYSTEM_PROMPT = """You are an M&A deal analyst assistant. You help investment bankers, 
PE professionals, and corporate M&A teams find information in deal documents.

RULES:
- Answer using ONLY the provided document excerpts
- Be precise with numbers, dates, and party names
- Cite your source as [Source N] after each factual claim
- If the information is not in the provided documents, say "This information is not available in the provided documents."
- Do not hallucinate or infer information not present in the text
- Use professional M&A terminology"""


def generate_answer(question: str, contexts: list[str]) -> str:
    """Produce a grounded answer from retrieved document chunks."""
    formatted_context = "\n\n".join(
        f"[Source {i+1}]: {ctx}" for i, ctx in enumerate(contexts)
    )
    user_prompt = f"""DOCUMENT EXCERPTS:
{formatted_context}

QUESTION: {question}

ANSWER:"""
    return call_llm_text(SYSTEM_PROMPT, user_prompt, temperature=LLM_TEMPERATURE)


def extract_structured(doc_chunks: list[str],
                        fields: list[str],
                        max_chunks: int = 15) -> dict:
    """Extract specific deal fields from document chunks as JSON. """
    context = "\n\n".join(doc_chunks[:max_chunks])
    system_prompt = f"""You extract structured data from M&A documents.
    Extract these fields: {', '.join(fields)}
    Return ONLY a valid JSON object. Use null for missing fields.
    Use the exact values found in the document — do not estimate."""

    data = call_llm_json(system_prompt, context, temperature=0.0)
    if data is None:
        return {"error": "Failed to get a valid JSON extraction from the LLM"}
    return data