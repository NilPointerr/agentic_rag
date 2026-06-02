## Agentic RAG

### Overview

Agentic RAG is a FastAPI-based retrieval-augmented generation (RAG) service that:
- **Ingests PDF documents**, chunks them, stores embeddings in **Pinecone**, and stores chunk text in a local **BM25** corpus.
- **Skips duplicate PDF content** during ingestion using exact normalized-content hashes for both whole documents and individual chunks.
- **Answers natural-language questions** with a hybrid retrieval pipeline: query expansion, BM25 search, vector search, RRF merge, cross-encoder reranking, then a **Groq LLM**.
- **Falls back to web search** when the internal document context is not sufficient.

The main HTTP API is exposed via FastAPI in `app.main:app`, with core logic implemented under the `app/` package.

### Tech Stack

- **Language**: Python (>= 3.12)
- **API framework**: FastAPI + Uvicorn
- **Vector store**: Pinecone
- **Lexical search**: local Okapi BM25 JSONL corpus
- **Hybrid merge**: Reciprocal Rank Fusion (RRF)
- **Reranking**: `sentence-transformers` cross encoder
- **LLM**: Groq (chat completions, tool calling)
- **Embeddings**: `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- **Environment config**: `pydantic-settings` with `.env`

### Project Structure (high level)

- `app/main.py` – FastAPI application entrypoint.
- `app/api/routes.py` – `/ingest` and `/query` endpoints.
- `frontend/` – Next.js frontend for interacting with the API.
- `app/ingestion/` – loading, chunking, embedding, and storing documents.
- `app/retriever/bm25_store.py` – local BM25 corpus storage and lexical scoring.
- `app/retriever/retriever.py` – hybrid retrieval, RRF merge, and reranking.
- `app/vectorstore/pinecone_client.py` – Pinecone client & index management.
- `app/llm/groq_client.py` – Groq client and chat completion wrapper.
- `app/agent/rag_agent.py` – agent orchestration + tool usage.
- `data/sample_docs/` – example documents.
- `data/uploads/` – uploaded PDFs are saved here.

---

### Prerequisites

- **Python**: 3.12 or later
- **Pinecone account & API key**
- **Groq account & API key**
- **uv** for dependency management.

### Environment Variables

Create a `.env` file in the project root (same level as `pyproject.toml`) with at least:

```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-rag-index-dimension-384  # optional, has this default

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3-32b                           # optional, default in code

EMBEDDING_MODEL=all-MiniLM-L6-v2                    # optional, default in code
EMBEDDING_DIMENSION=384                             # must match the model

RATE_LIMIT_ENABLED=false
TOP_K=5                                             # final chunks sent to the LLM
RETRIEVAL_CANDIDATES=20                             # candidates requested from BM25/vector search
RERANK_CANDIDATES=20                                # candidates sent to reranker
HYBRID_SEARCH_ENABLED=true
QUERY_EXPANSION_ENABLED=true
BM25_INDEX_PATH=data/bm25_chunks.jsonl
DEDUP_REGISTRY_PATH=data/dedup_registry.json
RRF_K=60

CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
```

These are read via `app/config/settings.py` using `pydantic-settings`.

---

### Setup

```bash
cd /home/dev62/Documents/agentic_rag
uv sync --group dev   # installs app + test dependencies from pyproject.toml / uv.lock
```

---

### Retrieval Pipeline

The internal document search flow is:

```text
User Query
    ↓
Query Expansion
    ↓
Hybrid Search
(BM25 + Vector)
    ↓
RRF Merge
    ↓
Re-ranking
    ↓
Top 5 Chunks
    ↓
LLM
```

How it works:

- **Query expansion** normalizes query terms for lexical search.
- **BM25 search** runs against the local JSONL corpus at `BM25_INDEX_PATH`.
- **Vector search** runs against Pinecone with the configured embedding model.
- **RRF merge** deduplicates chunks by stable `chunk_id` and combines BM25/vector rankings.
- **Reranking** uses the configured cross-encoder model when `RERANK_ENABLED=true`.
- The final response uses `TOP_K=5` chunks by default.

BM25 is built from chunk text during ingestion. If you already ingested documents before BM25 support was added, re-ingest those PDFs so their chunks are written to `data/bm25_chunks.jsonl`.

### Ingestion Pipeline

The `/ingest` flow is:

```text
Upload PDF
    ↓
Save file locally
    ↓
Extract PDF text
    ↓
Normalize + hash full document text
    ↓
Skip if document hash already exists
    ↓
Chunk pages
    ↓
Normalize + hash each chunk
    ↓
Reserve only unseen chunk hashes
    ↓
Embed and index only new chunks
    ↓
Persist document/chunk dedup status
```

Notes:

- Duplicate detection is based on the **normalized extracted PDF text**, not the PDF filename.
- Exact duplicate detection ignores casing and whitespace differences.
- Near-duplicate documents with small content edits are still treated as new content unless a chunk matches exactly after normalization.

### Running the API

From the project root:

```bash
cd /home/dev62/Documents/agentic_rag
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with plain `uvicorn` if installed globally/in your venv:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, you can access:
- **Interactive docs (Swagger)**: `http://localhost:8000/docs`
- **ReDoc docs**: `http://localhost:8000/redoc`

---

### Running The Next.js Frontend

Create a frontend environment file:

```bash
cd /home/dev62/Documents/agentic_rag/frontend
cp .env.local.example .env.local
```

Install dependencies:

```bash
cd /home/dev62/Documents/agentic_rag/frontend
npm install
```

Run the frontend:

```bash
cd /home/dev62/Documents/agentic_rag/frontend
npm run dev
```

Then open:
- **Next.js UI**: `http://localhost:3000`

The frontend expects the FastAPI backend to be running on `http://localhost:8000` by default. You can change that in `frontend/.env.local` by setting:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

### Running Backend And Frontend Together

Terminal 1:

```bash
cd /home/dev62/Documents/agentic_rag
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
cd /home/dev62/Documents/agentic_rag/frontend
npm run dev
```

Open:
- **Frontend UI**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`

---

### API Endpoints (summary)

- **POST** `/ingest`
  - **Description**: Upload a PDF, extract text, deduplicate by exact normalized content hash, embed only new chunks, store vectors in Pinecone, and upsert chunk text into the local BM25 corpus.
  - **Request**: `multipart/form-data` with field `file` (PDF only).
  - **Response**:
    ```json
    {
      "message": "PDF ingested successfully",
      "file_name": "example.pdf",
      "chunks_created": 8,
      "chunks_embedded": 5,
      "chunks_skipped_duplicate": 3,
      "pinecone_index": "agentic-rag-index-dimension-384",
      "skipped_duplicate": false,
      "duplicate_reason": null,
      "document_hash": "sha256-hash-of-normalized-document-text"
    }
    ```

- **POST** `/query`
  - **Description**: Ask a question; the agent retrieves internal context using hybrid search and uses Groq to generate an answer. Web search is used when internal context is insufficient.
  - **Request body**:
    ```json
    {
      "query": "Your question here"
    }
    ```
  - **Response**:
    ```json
    {
      "query": "Your question here",
      "answer": "Model-generated response...",
      "sources": [
        {
          "text": "Retrieved chunk text...",
          "score": 0.91,
          "vector_score": 0.82,
          "bm25_score": 3.14,
          "rrf_score": 0.03,
          "rerank_score": 0.91,
          "source_file": "example.pdf",
          "page_number": 2,
          "page_url": "/uploads/example.pdf#page=2",
          "chunk_index": 4,
          "chunk_id": "stable_chunk_id"
        }
      ],
      "images": []
    }
    ```

---

### Example Usage

#### 1. Ingest a PDF

Using `curl`:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/uploads/Chhatrapati-Shivaji.pdf"
```

Possible ingest outcomes:

- A brand-new PDF returns `chunks_embedded > 0`.
- A PDF with the same extracted text as an already indexed document returns `skipped_duplicate=true` and `duplicate_reason="document_hash_exists"`.
- A partially overlapping PDF may return both `chunks_embedded > 0` and `chunks_skipped_duplicate > 0`.

#### 2. Query the Agent

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who was Chhatrapati Shivaji?"
  }'
```

---

### Running Tests

Run the full test suite locally:

```bash
cd /home/dev62/Documents/agentic_rag
uv run --group dev pytest -q
```

Run the full test suite with coverage:

```bash
cd /home/dev62/Documents/agentic_rag
uv run --group dev pytest --cov=app --cov-report=term-missing
```

Coverage is configured to fail below `80%`.

The test suite is organized by app area:

- `tests/api/` for FastAPI route behavior
- `tests/ingestion/` for chunking, hashing, dedup store, and PDF loading
- `tests/retriever/` for BM25 and hybrid retrieval behavior
- `tests/search/` for web search adapters
- top-level `tests/test_*.py` files for cross-cutting modules like security, rate limiting, app wiring, and tool wrappers

The ingestion and dedup tests cover these scenarios:

- filename path traversal is sanitized before saving uploads
- document hashes are stable across casing and whitespace changes
- same content with different filenames is skipped as a duplicate document
- formatting-only text differences are skipped as duplicates
- same filename with different content is ingested as new content
- partially overlapping documents embed only the new chunks
- failed embeddings can be retried because `failed` dedup reservations do not permanently block later ingests
- scanned or image-only PDFs return a clear client-facing error

---

### Notes & Development

- The ingestion logic (loading, chunking, embedding) lives under `app/ingestion/`.
- BM25 storage is implemented in `app/retriever/bm25_store.py`.
- Hybrid retrieval is implemented in `app/retriever/retriever.py` and used inside `rag_agent`.
- The agent uses **tool calls** (hybrid internal search + web search) via Groq; you can customize tools in `app/llm_tools/llm_tools.py`.
- For local experimentation, you can modify or extend `rag_agent` in `app/agent/rag_agent.py`.
- `data/bm25_chunks.jsonl` and `data/dedup_registry.json` are generated runtime artifacts and are usually better kept out of Git.
