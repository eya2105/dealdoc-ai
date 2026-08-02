import json
import config  # ensure .env is loaded before OpenAI client
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=config.GROQ_API_KEY,
)

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

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""DOCUMENT EXCERPTS:
{formatted_context}

QUESTION: {question}

ANSWER:"""},
        ],
    )
    return response.choices[0].message.content


def extract_structured(doc_chunks: list[str],
                       fields: list[str],
                       max_chunks: int = 15) -> dict:
    """Extract specific deal fields from document chunks as JSON."""
    context = "\n\n".join(doc_chunks[:max_chunks])

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": f"""You extract structured data from M&A documents.
Extract these fields: {', '.join(fields)}
Return ONLY a valid JSON object. Use null for missing fields.
Use the exact values found in the document — do not estimate."""},
            {"role": "user", "content": context},
        ],
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse extraction",
                "raw": response.choices[0].message.content}
