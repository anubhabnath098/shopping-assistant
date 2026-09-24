from typing import List, Optional, Iterator
from core.interfaces.llm_provider import BaseLLMProvider
from core.models.schemas import RetrievedChunk, ConversationTurn, MultiModalQuery
from generation.prompt_builder import PromptBuilder


class ResponseGenerator:
    """Combines retrieved context, conversation history, and the query with the LLM.

    Two LLM providers are used: a vision-capable one for image queries
    (e.g. Gemini) and a text-only one for everything else (e.g. Qwen).
    """

    def __init__(
        self,
        text_llm_provider: BaseLLMProvider,
        vision_llm_provider: BaseLLMProvider,
        prompt_builder: PromptBuilder,
    ):
        self._text_llm_provider = text_llm_provider
        self._vision_llm_provider = vision_llm_provider
        self._prompt_builder = prompt_builder

    def _provider_for(self, query: MultiModalQuery) -> BaseLLMProvider:
        return self._vision_llm_provider if query.has_image else self._text_llm_provider

    def generate(
        self,
        query: MultiModalQuery,
        retrieved_chunks: List[RetrievedChunk],
        history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        prompt = self._prompt_builder.build(query, retrieved_chunks, history)
        image_path = query.image_path if query.has_image else None
        return self._provider_for(query).generate(prompt, image_path=image_path)

    def generate_stream(
        self,
        query: MultiModalQuery,
        retrieved_chunks: List[RetrievedChunk],
        history: Optional[List[ConversationTurn]] = None,
    ) -> Iterator[str]:
        prompt = self._prompt_builder.build(query, retrieved_chunks, history)
        image_path = query.image_path if query.has_image else None
        yield from self._provider_for(query).generate_stream(prompt, image_path=image_path)