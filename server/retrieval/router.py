from core.models.schemas import RouteDecision, RetrievalPath


class AdaptiveRouter:
    """
    AdaRAG-inspired dynamic router: decides between the light (fast) and
    heavy (accurate) retrieval paths based on the light path's confidence.
    """

    def __init__(self, confidence_threshold: float = 0.55):
        self._confidence_threshold = confidence_threshold

    def decide(self, light_confidence: float) -> RouteDecision:
        if light_confidence >= self._confidence_threshold:
            return RouteDecision(
                path=RetrievalPath.LIGHT,
                confidence=light_confidence,
                reason=f"confidence {light_confidence:.3f} >= threshold {self._confidence_threshold}",
            )
        return RouteDecision(
            path=RetrievalPath.HEAVY,
            confidence=light_confidence,
            reason=f"confidence {light_confidence:.3f} < threshold {self._confidence_threshold}",
        )