from typing import List, Optional, Iterator
from core.interfaces.llm_provider import BaseLLMProvider
from core.models.schemas import RetrievedChunk, ConversationTurn, MultiModalQuery
from generation.prompt_builder import PromptBuilder


class ResponseGenerator:
    """Combines retrieved context, conversation history, and the query with the LLM."""

    def __init__(self, llm_provider: BaseLLMProvider, prompt_builder: PromptBuilder):
        self._llm_provider = llm_provider
        self._prompt_builder = prompt_builder

    def generate(
        self,
        query: MultiModalQuery,
        retrieved_chunks: List[RetrievedChunk],
        history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        prompt = self._prompt_builder.build(query, retrieved_chunks, history)
        image_path = query.image_path if query.has_image else None
        return self._llm_provider.generate(prompt, image_path=image_path)

    def generate_stream(
        self,
        query: MultiModalQuery,
        retrieved_chunks: List[RetrievedChunk],
        history: Optional[List[ConversationTurn]] = None,
    ) -> Iterator[str]:
        prompt = self._prompt_builder.build(query, retrieved_chunks, history)
        image_path = query.image_path if query.has_image else None
        yield from self._llm_provider.generate_stream(prompt, image_path=image_path)