import os
import json
import time
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.database import get_connection, release_connection, init_db
from app.graph import init_graph, store_entities, get_document_entities, delete_document_graph
from app.auth import verify_api_key, rate_limiter
from app.models import (
    QueryRequest, QueryResponse, UploadResponse,
    EvaluatedQueryResponse, DocumentInfo, HealthResponse,
)
from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_chunks, embed_query
from app.ingestion.entity_extractor import extract_entities
from app.retrieval.vector_search import vector_search
from app.retrieval.bm25_search import bm25_search
from app.retrieval.graph_search import graph_retrieve
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.query_classifier import classify_query
from app.generation.llm import generate_answer
from app.memory.conversation import memory
from app.evaluation.ragas import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

query_cache = {}


def _init_background():
    try:
        init_db()
    except Exception as e:
        logger.error(f"PostgreSQL init failed: {e}")
    try:
        init_graph()
    except Exception as e:
        logger.error(f"Neo4j init failed (graph features disabled): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise Hybrid GraphRAG...")
    threading.Thread(target=_init_background, daemon=True).start()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Enterprise Hybrid GraphRAG",
    description="Hybrid retrieval combining vector search, BM25, and Neo4j knowledge graph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if request.url.path in ("/health", "/docs", "/openapi.json"):
        return await call_next(request)
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(rate_limiter.remaining(client_ip))
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed}ms)")
    return response


def validate_file(file: UploadFile, content: bytes):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max {settings.MAX_FILE_SIZE // 1024 // 1024}MB")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")


def insert_chunks(chunks: list[str], embeddings: list[list[float]], filename: str) -> int:
    conn = get_connection()
    inserted = 0
    try:
        cur = conn.cursor()
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = {"source": filename, "chunk_index": i}
            cur.execute(
                "INSERT INTO document_sections (content, meta, embedding) VALUES (%s, %s, %s)",
                (chunk, json.dumps(metadata), embedding),
            )
            inserted += 1
        conn.commit()
        cur.close()
        return inserted
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_connection(conn)


def log_search_analytics(question: str, retrieval_type: str, chunks_used: int, latency_ms: float):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO search_analytics (question, retrieval_type, chunks_used, latency_ms) VALUES (%s, %s, %s, %s)",
            (question, retrieval_type, chunks_used, latency_ms),
        )
        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
    finally:
        release_connection(conn)


def log_evaluation(question: str, answer: str, metrics: dict):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO rag_evaluations
            (question, answer, faithfulness, answer_relevance, context_precision, overall_score)
            VALUES (%s, %s, %s, %s, %s, %s)""",
            (question, answer, metrics["faithfulness"], metrics["answer_relevance"],
             metrics["context_precision"], metrics["overall_score"]),
        )
        conn.commit()
        cur.close()
    except Exception:
        conn.rollback()
    finally:
        release_connection(conn)


def retrieve(query: str, top_k: int = 5, use_graph: bool = True) -> tuple[list[dict], str]:
    classification = classify_query(query)
    retrieval_type = classification["type"]
    logger.info(f"Query classified as: {retrieval_type} ({classification['reason']})")

    all_results = []

    if retrieval_type in ("vector", "hybrid"):
        vec_results = vector_search(query, top_k=settings.VECTOR_TOP_K)
        all_results.append(vec_results)

    if retrieval_type in ("vector", "hybrid", "graph"):
        bm25_results = bm25_search(query, top_k=settings.BM25_TOP_K)
        all_results.append(bm25_results)

    if use_graph and retrieval_type in ("graph", "hybrid"):
        graph_results = graph_retrieve(query)
        if graph_results:
            all_results.append(graph_results)

    if not all_results:
        vec_results = vector_search(query, top_k=top_k)
        all_results.append(vec_results)

    fused = reciprocal_rank_fusion(all_results, k=settings.RRF_K)
    reranked = rerank(query, fused[:20], top_k=top_k)

    return reranked, retrieval_type


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(os.path.dirname(__file__), "..", "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Enterprise Hybrid GraphRAG</h1><p><a href='/docs'>API Docs</a></p>")


@app.get("/health", response_model=HealthResponse)
def health():
    pg_status = "connected"
    neo4j_status = "connected"
    try:
        conn = get_connection()
        conn.cursor().execute("SELECT 1")
        release_connection(conn)
    except Exception:
        pg_status = "disconnected"
    try:
        from app.graph import get_driver
        get_driver().verify_connectivity()
    except Exception:
        neo4j_status = "disconnected"
    return HealthResponse(
        status="healthy" if pg_status == "connected" else "degraded",
        postgres=pg_status,
        neo4j=neo4j_status,
        cache_size=len(query_cache),
    )


@app.get("/documents", response_model=list[DocumentInfo], dependencies=[Depends(verify_api_key)])
def list_documents():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT meta->>'source' AS source, COUNT(*) AS chunks
            FROM document_sections
            WHERE meta IS NOT NULL
            GROUP BY meta->>'source'
            ORDER BY source;
        """)
        rows = cur.fetchall()
        cur.close()

        results = []
        for row in rows:
            filename = row[0]
            chunk_count = row[1]
            try:
                entities = get_document_entities(filename)
            except Exception:
                entities = []
            results.append(DocumentInfo(
                filename=filename,
                chunks=chunk_count,
                entities=len(entities),
                uploaded_at=None,
            ))
        return results
    finally:
        release_connection(conn)


@app.post("/upload", response_model=UploadResponse, dependencies=[Depends(verify_api_key)])
async def upload_pdf(file: UploadFile = File(...), force: bool = False):
    content = await file.read()
    validate_file(file, content)

    if not force:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_sections WHERE meta->>'source' = %s", (file.filename,))
            count = cur.fetchone()[0]
            cur.close()
            if count > 0:
                raise HTTPException(status_code=409, detail=f"'{file.filename}' already ingested. Use ?force=true to re-upload.")
        finally:
            release_connection(conn)

    if force:
        conn = get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM document_sections WHERE meta->>'source' = %s", (file.filename,))
            conn.commit()
            cur.close()
        finally:
            release_connection(conn)
        try:
            delete_document_graph(file.filename)
        except Exception:
            pass

    text = extract_text_from_pdf(content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    chunks = chunk_text(text)
    if len(chunks) > settings.MAX_CHUNKS:
        raise HTTPException(status_code=400, detail=f"PDF too large — {len(chunks)} chunks, max {settings.MAX_CHUNKS}")

    embeddings = embed_chunks(chunks)
    inserted = insert_chunks(chunks, embeddings, file.filename)

    entities_data = {"entities": [], "relationships": []}
    try:
        full_text = " ".join(chunks[:20])
        entities_data = extract_entities(full_text)
        if entities_data["entities"]:
            store_entities(file.filename, entities_data["entities"], entities_data["relationships"])
    except Exception as e:
        logger.error(f"Entity extraction failed for '{file.filename}': {e}")

    query_cache.clear()

    return UploadResponse(
        filename=file.filename,
        chunks_inserted=inserted,
        entities_extracted=len(entities_data["entities"]),
        relationships_found=len(entities_data["relationships"]),
        message=f"Successfully ingested '{file.filename}'",
    )


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cache_key = f"{request.question.strip().lower()}::{request.top_k}::{request.use_graph}"
    if cache_key in query_cache:
        return QueryResponse(**query_cache[cache_key])

    start = time.time()
    chunks, retrieval_type = retrieve(request.question, request.top_k, request.use_graph)
    latency_ms = (time.time() - start) * 1000

    history = []
    if request.session_id:
        history = memory.get_history(request.session_id)

    answer = generate_answer(request.question, chunks, history)

    if request.session_id:
        memory.add_message(request.session_id, "user", request.question)
        memory.add_message(request.session_id, "assistant", answer)

    sources = []
    for chunk in chunks:
        meta = chunk.get("meta")
        if isinstance(meta, dict) and meta.get("source"):
            if meta["source"] not in sources:
                sources.append(meta["source"])

    result = {
        "question": request.question,
        "answer": answer,
        "chunks_used": len(chunks),
        "retrieval_type": retrieval_type,
        "sources": sources,
    }
    query_cache[cache_key] = result

    log_search_analytics(request.question, retrieval_type, len(chunks), latency_ms)

    return QueryResponse(**result)


@app.post("/evaluate-query", response_model=EvaluatedQueryResponse, dependencies=[Depends(verify_api_key)])
async def evaluate_query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks, retrieval_type = retrieve(request.question, request.top_k, request.use_graph)
    answer = generate_answer(request.question, chunks)

    chunks_text = [c["content"] for c in chunks]
    metrics = evaluate(request.question, answer, chunks_text)
    log_evaluation(request.question, answer, metrics)

    return EvaluatedQueryResponse(
        question=request.question,
        answer=answer,
        chunks_used=len(chunks),
        retrieval_type=retrieval_type,
        **metrics,
    )


@app.post("/memory/{session_id}/clear", dependencies=[Depends(verify_api_key)])
def clear_memory(session_id: str):
    memory.clear(session_id)
    return {"message": f"Memory cleared for session {session_id}"}


@app.post("/cache/clear", dependencies=[Depends(verify_api_key)])
def clear_cache():
    count = len(query_cache)
    query_cache.clear()
    return {"message": f"Cache cleared. {count} entries removed."}


@app.get("/analytics", dependencies=[Depends(verify_api_key)])
def get_analytics():
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as total_queries,
                AVG(latency_ms) as avg_latency,
                retrieval_type,
                COUNT(*) as count
            FROM search_analytics
            GROUP BY retrieval_type
        """)
        rows = cur.fetchall()
        cur.close()

        total = sum(r[3] for r in rows)
        by_type = {r[2]: {"count": r[3], "percentage": round(r[3] / total * 100, 1) if total else 0} for r in rows}

        cur = conn.cursor()
        cur.execute("SELECT AVG(latency_ms) FROM search_analytics")
        avg_latency = cur.fetchone()[0]
        cur.close()

        return {
            "total_queries": total,
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else 0,
            "by_retrieval_type": by_type,
        }
    finally:
        release_connection(conn)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
