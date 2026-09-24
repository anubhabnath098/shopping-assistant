from typing import Optional, Iterator
from PIL import Image
import google.generativeai as genai
from core.interfaces.llm_provider import BaseLLMProvider


class GeminiLLMProvider(BaseLLMProvider):
    """Gemini-backed generator. Swap for another BaseLLMProvider via the factory."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
    ):
        if not api_key:
            raise ValueError("Gemini API key is required (set GEMINI_API_KEY in .env)")
        genai.configure(api_key=api_key)
        self._model_name = model_name
        self._model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
        )

    def generate(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
        try:
            parts = self._build_parts(prompt, image_path)
            response = self._model.generate_content(parts)
            return response.text.strip() if response and response.text else "I could not generate a response."
        except Exception as exc:
            return f"[LLM generation error]: {exc}"

    def generate_stream(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> Iterator[str]:
        try:
            parts = self._build_parts(prompt, image_path)
            response_stream = self._model.generate_content(parts, stream=True)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            yield f"[LLM generation error]: {exc}"

    @staticmethod
    def _build_parts(prompt: str, image_path: Optional[str]):
        parts = [prompt]
        if image_path:
            parts.append(Image.open(image_path).convert("RGB"))
        return parts