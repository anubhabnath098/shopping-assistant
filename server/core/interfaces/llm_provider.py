from abc import ABC, abstractmethod
from typing import Optional, Iterator


class BaseLLMProvider(ABC):
    """Contract for any generation backend (Gemini, OpenAI, local LLM, ...)."""

    @abstractmethod
    def generate(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, prompt: str, image_path: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Yields the answer incrementally, chunk by chunk, as the model produces it."""
        raise NotImplementedError