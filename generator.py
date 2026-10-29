from config import LLM_TEMPERATURE
from ports import LLM

SYSTEM_PROMPT = """You are an M&A deal analyst assistant. You help investment bankers, 
PE professionals, and corporate M&A teams find information in deal documents.

RULES:
- Answer using ONLY the provided document excerpts
- Be precise with numbers, dates, and party names
- Cite your source as [Source N] after each factual claim
- If the information is not in the provided documents, say "This information is not available in the provided documents."
- Do not hallucinate or infer information not present in the text
- Use professional M&A terminology"""


class AnswerGenerator:
    """Produces a grounded, cited answer from retrieved chunks."""

    def __init__(self, llm: LLM, temperature: float = LLM_TEMPERATURE):
        self._llm = llm
        self._temperature = temperature

    def answer(self, question: str, contexts: list[str]) -> str:
        formatted_context = "\n\n".join(f"[Source {i}]: {ctx}" for i, ctx in enumerate(contexts, start=1))
        user_prompt = f"""DOCUMENT EXCERPTS:
{formatted_context}

QUESTION: {question}

ANSWER:"""
        return self._llm.complete_text(SYSTEM_PROMPT, user_prompt, temperature=self._temperature)
