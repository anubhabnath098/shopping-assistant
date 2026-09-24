from typing import List
from core.interfaces.retriever import BaseRetriever
from core.models.schemas import RetrievedChunk, MultiModalQuery
from retrieval.dual_index_retriever import DualIndexRetriever


class LightRetriever(BaseRetriever):
    """
    AdaRAG 'light' path: a single fast ANN pass over the dual (text+image)
    index — used for high-confidence, easy queries, targeting low latency.
    """

    def __init__(self, dense_retriever: DualIndexRetriever):
        self._dense_retriever = dense_retriever

    def retrieve(self, query: MultiModalQuery, top_k: int) -> List[RetrievedChunk]:
        return self._dense_retriever.retrieve(query, top_k)

    def confidence(self, query: MultiModalQuery) -> float:
        return self._dense_retriever.top_confidence(query)

    def dense_retriever_text_size(self) -> int:
        return self._dense_retriever.text_index_size()

    def dense_retriever_image_size(self) -> int:
        return self._dense_retriever.image_index_size()