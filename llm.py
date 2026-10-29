"""LLM adapter for Groq's OpenAI-compatible API (implements the LLM port)."""
import json
import logging
from typing import Optional

from openai import OpenAI

from config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MAX_RETRIES, LLM_MODEL

logger = logging.getLogger(__name__)


class GroqLLM:
    def __init__(self, api_key: Optional[str] = GROQ_API_KEY, base_url: str = GROQ_BASE_URL,
                 model: str = LLM_MODEL, max_retries: int = LLM_MAX_RETRIES, client=None):
        self._api_key, self._base_url, self._model, self._max_retries = api_key, base_url, model, max_retries
        self._client = client  # injectable for tests

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file before calling the LLM.")
            # on a 429 (rate limit) the SDK waits Retry-After / backs off, then retries
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key, max_retries=self._max_retries)
        return self._client

    def _create(self, system: str, user: str, temperature: float, **options):
        return self._get_client().chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **options,
        )

    def complete_text(self, system: str, user: str, temperature: float = 0.1) -> str:
        return self._create(system, user, temperature).choices[0].message.content

    def complete_json(self, system: str, user: str, temperature: float = 0.0) -> Optional[dict]:
        """A parsed JSON object, or None on any failure (logged)."""
        try:
            response = self._create(system, user, temperature, response_format={"type": "json_object"})
            return json.loads(response.choices[0].message.content)
        except Exception:
            logger.exception("complete_json failed")
            return None
