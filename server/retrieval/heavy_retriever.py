from typing import List
from core.interfaces.retriever import BaseRetriever
from core.interfaces.reranker import BaseReranker
from core.models.schemas import RetrievedChunk, MultiModalQuery
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dual_index_retriever import DualIndexRetriever


class HeavyRetriever(BaseRetriever):
    """
    AdaRAG 'heavy' path: hybrid sparse (BM25) + dense (dual-index) search,
    fused through a reranker — used for low-confidence / hard queries.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DualIndexRetriever,
        reranker: BaseReranker,
        pool_size: int = 20,
    ):
        self._bm25_retriever = bm25_retriever
        self._dense_retriever = dense_retriever
        self._reranker = reranker
        self._pool_size = pool_size

    def retrieve(self, query: MultiModalQuery, top_k: int) -> List[RetrievedChunk]:
        sparse_hits = self._bm25_retriever.retrieve(query, self._pool_size)
        dense_hits = self._dense_retriever.retrieve(query, self._pool_size)
        candidate_pool = sparse_hits + dense_hits
        return self._reranker.rerank(query, candidate_pool, top_k)