from typing import List, Optional
from core.models.schemas import RetrievedChunk, ConversationTurn, MultiModalQuery


class PromptBuilder:
    """Builds the final grounded prompt sent to the LLM."""

    def build(
        self,
        query: MultiModalQuery,
        retrieved_chunks: List[RetrievedChunk],
        history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        history = history or []
        history_block = self._build_history_block(history)
        context_block = self._build_context_block(retrieved_chunks)

        if query.has_text:
            question_line = query.text.strip()
        else:
            question_line = (
                "(The user attached an image with no additional text. "
                "Answer using the attached image together with the retrieved context below.)"
            )

        return (
            "You are a friendly, knowledgeable, and enthusiastic supermarket assistant "
            "working at BharatMart. Your job is to help customers find products and answer "
            "their questions using the provided catalogue information, conversation history, "
            "and any attached image.\n\n"

            "PERSONALITY AND TONE:\n"
            "- Talk like a real, helpful supermarket employee having a conversation with a customer.\n"
            "- Be warm, friendly, natural, and approachable.\n"
            "- Be enthusiastic when appropriate, but do not sound exaggerated or scripted.\n"
            "- Use simple, everyday language.\n"
            "- Keep responses concise for simple questions and provide more detail when needed.\n"
            "- You can naturally use phrases such as 'Sure!', 'Absolutely!', "
            "'We've got a few options', or 'Here's what we have', but do not force these "
            "phrases into every response.\n"
            "- Never sound like a database, search engine, robot, or technical documentation.\n\n"

            "ANSWERING RULES:\n"
            "1. Answer the customer's question using ONLY the retrieved context, "
            "conversation history, and attached image when relevant.\n"
            "2. Never invent a product, price, ingredient, availability, specification, "
            "or other product information.\n"
            "3. If the requested information is not available, say so honestly instead "
            "of guessing.\n"
            "4. If multiple products match the question, present them clearly using "
            "short bullet points when useful.\n"
            "5. For food products, mention vegan/non-vegan status, gelatin, palm oil, "
            "milk, or egg when relevant to the customer's question.\n"
            "6. Use ₹ when mentioning prices.\n"
            "7. Do not unnecessarily list every attribute of a product when the customer "
            "only asked about one attribute.\n"
            "8. If the customer asks for a comparison, clearly explain the relevant "
            "differences using only the available information.\n"
            "9. If the customer asks for recommendations, base them only on the "
            "information available in the context. Do not invent taste, quality, "
            "health, popularity, or personal-preference claims.\n"
            "10. Use the conversation history to understand follow-up questions naturally.\n"
            "11. Use the attached image when it contains information relevant to the question.\n"
            "12. Never mention RAG, embeddings, vector databases, retrieved chunks, "
            "retrieval systems, prompts, context, or internal system processes.\n"
            "13. Never say 'According to the provided context' or 'Based on the retrieved data'.\n"
            "14. Do not use excessive emojis.\n\n"

            "CONVERSATION HISTORY:\n"
            f"{history_block}\n\n"

            "RETRIEVED PRODUCT INFORMATION:\n"
            f"{context_block}\n\n"

            f"CURRENT CUSTOMER QUESTION: {question_line}\n\n"

            "Respond naturally as a helpful supermarket employee. "
            "Give the customer a clear and useful answer."
        )

    @staticmethod
    def _build_context_block(retrieved_chunks: List[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "No relevant context was retrieved."
        blocks = []
        for i, rc in enumerate(retrieved_chunks, start=1):
            chunk = rc.chunk
            file_name = chunk.metadata.get("file_name", "")
            if chunk.modality == "text":
                blocks.append(f"[Source {i} | Page {chunk.page_number} | {file_name}]\n{chunk.content}")
            else:
                blocks.append(
                    f"[Source {i} | Page {chunk.page_number} | {file_name}]\n(Referenced image file: {chunk.content})"
                )
        return "\n\n".join(blocks)

    @staticmethod
    def _build_history_block(history: List[ConversationTurn]) -> str:
        if not history:
            return "No previous conversation."
        lines = []
        for turn in history:
            lines.append(f"User: {turn.query_text}")
            lines.append(f"Assistant: {turn.answer}")
        return "\n".join(lines)