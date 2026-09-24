from typing import Optional, Iterator
from openai import OpenAI
from core.interfaces.llm_provider import BaseLLMProvider


class GroqLLMProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str = "openai/gpt-oss-20b",
                 temperature: float = 0.3, max_output_tokens: int = 1024, **_):
        if not api_key:
            raise ValueError("Groq API key is required")
        self._client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=30, max_retries=1)
        self._model = model_name
        self._temperature = temperature
        self._max_tokens = max_output_tokens

    def _messages(self, prompt):
        return [{"role": "user", "content": prompt}]  # image_path ignored (text-only)

    def generate(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
        r = self._client.chat.completions.create(
            model=self._model, messages=self._messages(prompt),
            temperature=self._temperature, max_tokens=self._max_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    def generate_stream(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model, messages=self._messages(prompt),
            temperature=self._temperature, max_tokens=self._max_tokens, stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content