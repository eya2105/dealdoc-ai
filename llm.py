"""LLM client — the only module that talks to Groq directly."""
import json
import logging

from openai import OpenAI
from config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)

client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)


def call_llm_text(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """Call the LLM and return plain text content."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict | None:
    """Call the LLM expecting a JSON object back."""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        logger.exception("call_llm_json failed")
        return None