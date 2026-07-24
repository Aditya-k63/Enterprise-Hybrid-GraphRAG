import logging

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Loaded cross-encoder reranker")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, item["content"]) for item in candidates]
    scores = reranker.predict(pairs)

    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
