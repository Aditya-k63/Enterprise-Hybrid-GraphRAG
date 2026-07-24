import logging

logger = logging.getLogger(__name__)


def cosine_similarity(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_faithfulness(answer: str, chunks: list[str]) -> float:
    no_info = ["i don't have enough", "i cannot answer", "not in the context"]
    if any(phrase in answer.lower() for phrase in no_info):
        return 1.0

    from app.ingestion.embedder import get_embedder
    model = get_embedder()
    answer_emb = model.encode(answer)
    context = " ".join(chunks)
    context_emb = model.encode(context)
    return round(cosine_similarity(answer_emb, context_emb), 4)


def compute_answer_relevance(question: str, answer: str) -> float:
    from app.ingestion.embedder import get_embedder
    model = get_embedder()
    q_emb = model.encode(question)
    a_emb = model.encode(answer)
    return round(cosine_similarity(q_emb, a_emb), 4)


def compute_context_precision(question: str, chunks: list[str], threshold: float = 0.3) -> float:
    if not chunks:
        return 0.0
    from app.ingestion.embedder import get_embedder
    model = get_embedder()
    q_emb = model.encode(question)
    chunk_embs = model.encode(chunks)
    relevant = sum(1 for emb in chunk_embs if cosine_similarity(q_emb, emb) >= threshold)
    return round(relevant / len(chunks), 4)


def evaluate(question: str, answer: str, chunks: list[str]) -> dict:
    faithfulness = compute_faithfulness(answer, chunks)
    relevance = compute_answer_relevance(question, answer)
    precision = compute_context_precision(question, chunks)
    overall = round(
        (faithfulness * 0.4) + (relevance * 0.4) + (precision * 0.2), 4
    )
    return {
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "context_precision": precision,
        "overall_score": overall,
    }
