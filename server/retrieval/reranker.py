from typing import List, Dict
from core.interfaces.reranker import BaseReranker
from core.models.schemas import RetrievedChunk, MultiModalQuery


class ReciprocalRankFusionReranker(BaseReranker):
    """
    Fuses multiple ranked candidate lists (BM25 + dense) into a single
    ranking using Reciprocal Rank Fusion — the 'multimodal re-ranker' step
    of the AdaRAG heavy path.
    """

    def __init__(self, k_constant: int = 60):
        self._k_constant = k_constant

    def rerank(self, query: MultiModalQuery, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        grouped: Dict[str, List[RetrievedChunk]] = {}
        for rc in candidates:
            grouped.setdefault(rc.source_retriever, []).append(rc)

        fused_scores: Dict[str, float] = {}
        best_chunk_by_id: Dict[str, RetrievedChunk] = {}
        for _, items in grouped.items():
            ranked = sorted(items, key=lambda rc: rc.score, reverse=True)
            for rank, rc in enumerate(ranked):
                cid = rc.chunk.chunk_id
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (self._k_constant + rank + 1)
                best_chunk_by_id[cid] = rc

        ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=best_chunk_by_id[cid].chunk, score=fused_scores[cid], source_retriever="reranked")
            for cid in ranked_ids
        ]