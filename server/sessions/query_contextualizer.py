import logging
import re
from typing import List
from core.interfaces.llm_provider import BaseLLMProvider
from core.models.schemas import ConversationTurn

log = logging.getLogger("rag.contextualizer")


class QueryContextualizer:
    """
    Rewrites a possibly elliptical follow-up ('what about its price?') into a
    standalone query using prior conversation turns, so retrieval always
    operates on a self-contained question instead of a fragment.
    Any LLM failure or unusable output falls back to the original text.
    """

    MAX_REWRITE_CHARS = 400

    def __init__(self, llm_provider: BaseLLMProvider):
        self._llm_provider = llm_provider

    def contextualize(self, current_text: str, history: List[ConversationTurn]) -> str:
        if not history or not current_text:
            return current_text

        history_lines = []
        for turn in history:
            who = "User (shared an image)" if turn.query_image_path else "User"
            history_lines.append(f"{who}: {turn.query_text}")
            history_lines.append(f"Assistant: {turn.answer}")
        history_block = "\n".join(history_lines)

        prompt = (
            "Given the conversation history and a follow-up question, rewrite the follow-up "
            "into a standalone question that contains all necessary context from the history. "
            "If the customer earlier shared an image, the assistant's reply describes it, so "
            "use the product it identified. "
            "If the follow-up question is already standalone, return it unchanged. "
            "Only output the rewritten question, nothing else — no preamble.\n\n"
            f"CONVERSATION HISTORY:\n{history_block}\n\n"
            f"FOLLOW-UP QUESTION: {current_text}\n\n"
            "STANDALONE QUESTION:"
        )

        try:
            raw = self._llm_provider.generate(prompt)
        except Exception as exc:
            log.warning("contextualizer LLM failed (%s); using original query", exc)
            return current_text

        return self._clean(raw, current_text)

    def _clean(self, raw: str, fallback: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
        if not text or text.startswith("[LLM generation error]"):
            return fallback
        text = text.splitlines()[0].strip().strip('"').strip()
        if not text or len(text) > self.MAX_REWRITE_CHARS:
            return fallback
        return text