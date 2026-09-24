from typing import List
from core.interfaces.llm_provider import BaseLLMProvider
from core.models.schemas import ConversationTurn


class QueryContextualizer:
    """
    Rewrites a possibly elliptical follow-up ('what about its price?') into a
    standalone query using prior conversation turns, so retrieval always
    operates on a self-contained question instead of a fragment.
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        self._llm_provider = llm_provider

    def contextualize(self, current_text: str, history: List[ConversationTurn]) -> str:
        if not history or not current_text:
            return current_text

        history_lines = []
        for turn in history:
            history_lines.append(f"User: {turn.query_text}")
            history_lines.append(f"Assistant: {turn.answer}")
        history_block = "\n".join(history_lines)

        prompt = (
            "Given the conversation history and a follow-up question, rewrite the follow-up "
            "into a standalone question that contains all necessary context from the history. "
            "If the follow-up question is already standalone, return it unchanged. "
            "Only output the rewritten question, nothing else — no preamble.\n\n"
            f"CONVERSATION HISTORY:\n{history_block}\n\n"
            f"FOLLOW-UP QUESTION: {current_text}\n\n"
            "STANDALONE QUESTION:"
        )
        rewritten = self._llm_provider.generate(prompt)
        return rewritten.strip() or current_text