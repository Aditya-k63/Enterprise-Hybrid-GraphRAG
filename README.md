# Enterprise Hybrid GraphRAG

A production-grade Retrieval-Augmented Generation system that combines **vector search**, **BM25 keyword search**, and **Neo4j knowledge graph traversal** for accurate, grounded answers from any PDF document.

---

## Live Demo

Upload any PDF, ask questions, and get answers grounded in the actual document with source citations. The system automatically extracts entities and relationships into a knowledge graph for multi-hop reasoning.

---

## How It Works

```
Upload PDF
    ↓
Text extraction → Semantic chunking → Embedding generation
    ↓                                    ↓
Store in PostgreSQL (pgvector)    Extract entities → Store in Neo4j
    ↓                                    ↓
Ask a question
    ↓
Query classifier determines retrieval strategy:
├─ Vector: semantic similarity search
├─ BM25: keyword matching
├─ Graph: entity relationship traversal
└─ Hybrid: combines all three
    ↓
Reciprocal Rank Fusion merges results
    ↓
Cross-encoder reranker picks best chunks
    ↓
LLM generates grounded answer with citations
```

---

## Results

| Feature | Status |
|---|---|
| PDF ingestion | ✅ |
| Semantic chunking | ✅ |
| Embedding generation | ✅ |
| PostgreSQL + pgvector | ✅ |
| Neo4j knowledge graph | ✅ |
| BM25 keyword search | ✅ |
| Hybrid retrieval (RRF) | ✅ |
| Cross-encoder reranker | ✅ |
| Query classifier | ✅ |
| Multi-hop graph traversal | ✅ |
| LLM answer generation | ✅ |
| Source citations | ✅ |
| Conversation memory | ✅ |
| RAGAS evaluation | ✅ |
| JWT authentication | ✅ |
| Rate limiting | ✅ |
| Redis-free caching | ✅ |
| Search analytics | ✅ |
| Docker Compose | ✅ |
| Tests | ✅ |
| CI/CD | ✅ |

---

## Tech Stack

| Layer | Tool |
|---|---|
| Vector DB | pgvector (PostgreSQL) |
| Knowledge Graph | Neo4j |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| LLM | Groq (llama-3.1-8b-instant) |
| Backend | FastAPI |
| Frontend | Streamlit + Built-in HTML UI |
| Containerization | Docker Compose |

---

## Project Structure

```
Enterprise-Hybrid-GraphRAG/
├── app/
│   ├── main.py              # FastAPI app + routes
│   ├── config.py            # Settings
│   ├── database.py          # PostgreSQL connection pool
│   ├── graph.py             # Neo4j operations
│   ├── auth.py              # API key auth + rate limiting
│   ├── models.py            # Pydantic schemas
│   ├── ingestion/
│   │   ├── pdf_parser.py    # PDF text extraction
│   │   ├── chunker.py       # Semantic chunking
│   │   ├── embedder.py      # Embedding generation
│   │   └── entity_extractor.py  # NER via LLM
│   ├── retrieval/
│   │   ├── vector_search.py │   │   ├── bm25_search.py
│   │   ├── graph_search.py  # Neo4j traversal
│   │   ├── hybrid.py        # Reciprocal rank fusion
│   │   ├── reranker.py      # Cross-encoder reranking
│   │   └── query_classifier.py
│   ├── generation/
│   │   └── llm.py           # Groq LLM integration
│   ├── memory/
│   │   └── conversation.py  # Session-based chat memory
│   └── evaluation/
│       └── ragas.py         # Faithfulness, relevance, precision
├── tests/
├── frontend/
│   └── app.py               # Streamlit UI
├── static/
│   └── index.html           # Built-in web UI
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .github/workflows/
    └── ci.yml
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector
- Neo4j 5+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Quick Start

```bash
git clone https://github.com/Aditya-k63/Enterprise-Hybrid-GraphRAG.git
cd Enterprise-Hybrid-GraphRAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

### Database Setup

**PostgreSQL:**
```sql
CREATE DATABASE graphrag;
CREATE EXTENSION IF NOT EXISTS vector;
```

**Neo4j:** Run `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5-community`

### Run

```bash
# API
uvicorn app.main:app --reload

# Streamlit (separate terminal)
streamlit run frontend/app.py
```

### Docker Compose

```bash
docker-compose up --build
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | System health check |
| GET | `/documents` | List ingested PDFs |
| POST | `/upload` | Upload PDF (max 10MB) |
| POST | `/query` | Ask a question |
| POST | `/evaluate-query` | Ask + get quality scores |
| GET | `/analytics` | Search analytics |
| POST | `/memory/{session_id}/clear` | Clear conversation memory |
| POST | `/cache/clear` | Clear query cache |

All endpoints except `/` and `/health` require `X-API-Key` header.

---

## Evaluation

The system evaluates every `/evaluate-query` response on three metrics:

- **Faithfulness** (0-1): Is the answer grounded in the retrieved chunks?
- **Answer Relevance** (0-1): Does it actually answer the question?
- **Context Precision** (0-1): Were the retrieved chunks useful?

Results are logged to PostgreSQL for tracking over time.

---

## Architecture Decisions

**Why hybrid retrieval?**
Vector search catches semantic meaning. BM25 catches exact keywords, names, and dates. Graph traversal catches entity relationships that neither vector nor BM25 can find. Combining all three with RRF gives the best of each world.

**Why Neo4j?**
Knowledge graphs store explicit relationships between entities. When someone asks "Who founded the company that acquired X?", the graph can traverse: X → acquired_by → Company → founded_by → Person. Vector search alone can't do this reliably.

**Why query classification?**
Not every query needs graph traversal. "What is machine learning?" is a vector search question. "How does Company A relate to Company B?" needs the graph. Classifying first saves latency and improves accuracy.

---

> Built by [Aditya Kumar](https://github.com/Aditya-k63) as part of an ML portfolio project.
