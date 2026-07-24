import logging

logger = logging.getLogger(__name__)

_connection_pool = None


def _get_conn_kwargs():
    from app.config import settings
    return {
        "dbname": settings.DB_NAME,
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
        "host": settings.DB_HOST,
        "port": settings.DB_PORT,
        "sslmode": "require",
    }


def get_pool():
    global _connection_pool
    if _connection_pool is None:
        from psycopg2 import pool
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=5,
            **_get_conn_kwargs(),
        )
        logger.info("PostgreSQL connection pool created")
    return _connection_pool


def get_connection():
    from pgvector.psycopg2 import register_vector
    conn = get_pool().getconn()
    register_vector(conn)
    return conn


def release_connection(conn):
    get_pool().putconn(conn)


def init_db():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_sections (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                meta JSONB,
                embedding VECTOR(384)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                chunk_count INT DEFAULT 0,
                entity_count INT DEFAULT 0,
                uploaded_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_evaluations (
                id BIGSERIAL PRIMARY KEY,
                question TEXT,
                answer TEXT,
                faithfulness FLOAT,
                answer_relevance FLOAT,
                context_precision FLOAT,
                overall_score FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS search_analytics (
                id BIGSERIAL PRIMARY KEY,
                question TEXT,
                retrieval_type VARCHAR(20),
                chunks_used INT,
                latency_ms FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding
                ON document_sections
                USING hnsw (embedding vector_cosine_ops);
            """)
        except Exception:
            conn.rollback()
        conn.commit()
        cur.close()
        logger.info("PostgreSQL tables initialized")
    finally:
        release_connection(conn)
