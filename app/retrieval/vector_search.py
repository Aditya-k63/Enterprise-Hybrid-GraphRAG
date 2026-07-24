import logging

logger = logging.getLogger(__name__)


def vector_search(query: str, top_k: int = 10) -> list[dict]:
    from app.database import get_connection, release_connection
    from app.ingestion.embedder import embed_query

    query_embedding = embed_query(query)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, content, meta, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_sections
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_embedding, top_k))
        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0],
                "content": row[1],
                "meta": row[2],
                "score": float(row[3]),
                "source": "vector",
            })
        cur.close()
        return results
    finally:
        release_connection(conn)
