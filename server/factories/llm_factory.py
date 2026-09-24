from typing import Dict, Type
from core.interfaces.llm_provider import BaseLLMProvider
from llms.gemini_llm import GeminiLLMProvider


class LLMFactory:
    _registry: Dict[str, Type[BaseLLMProvider]] = {
        "gemini": GeminiLLMProvider,
    }

    @classmethod
    def register(cls, key: str, llm_cls: Type[BaseLLMProvider]) -> None:
        cls._registry[key] = llm_cls

    @classmethod
    def create(cls, llm_type: str, **kwargs) -> BaseLLMProvider:
        llm_cls = cls._registry.get(llm_type.lower())
        if llm_cls is None:
            raise ValueError(f"Unknown LLM type: {llm_type}. Available: {list(cls._registry)}")
        return llm_cls(**kwargs)