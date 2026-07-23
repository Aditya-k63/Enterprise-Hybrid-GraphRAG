import logging
import numpy as np
from rank_bm25 import BM25Okapi
from app.database import get_connection, release_connection

logger = logging.getLogger(__name__)


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, content, meta FROM document_sections;")
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return []

        contents = [row[1] for row in rows]
        tokenized_corpus = [doc.lower().split() for doc in contents]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": rows[idx][0],
                    "content": rows[idx][1],
                    "meta": rows[idx][2],
                    "score": float(scores[idx]),
                    "source": "bm25",
                })
        return results
    finally:
        release_connection(conn)
