## Agentic RAG

### Overview

Agentic RAG is a FastAPI-based retrieval-augmented generation (RAG) service that:
- **Ingests PDF documents**, chunks them, and stores embeddings in **Pinecone**.
- **Answers natural-language questions** by retrieving relevant context and calling a **Groq LLM** with tool usage (vector search + web search).

The main HTTP API is exposed via FastAPI in `app.main:app`, with core logic implemented under the `app/` package.

### Tech Stack

- **Language**: Python (>= 3.12)
- **API framework**: FastAPI + Uvicorn
- **Vector store**: Pinecone
- **LLM**: Groq (chat completions, tool calling)
- **Embeddings**: `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- **Environment config**: `pydantic-settings` with `.env`

### Project Structure (high level)

- `app/main.py` – FastAPI application entrypoint.
- `app/api/routes.py` – `/ingest` and `/query` endpoints.
- `app/ingestion/` – loading, chunking, embedding, and storing documents.
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
- (Recommended) **uv** (`pip install uv`) for dependency management, or use classic `pip`.

### Environment Variables

Create a `.env` file in the project root (same level as `pyproject.toml`) with at least:

```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-rag-index-dimension-384  # optional, has this default

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b                       # optional, default in code

EMBEDDING_MODEL=all-MiniLM-L6-v2                    # optional, default in code
EMBEDDING_DIMENSION=384                             # must match the model
TOP_K=3                                             # optional
SIMILARITY_THRESHOLD=0.65                           # optional
```

These are read via `app/config/settings.py` using `pydantic-settings`.

---

### Setup

#### Option 1: Using `uv` (recommended)

```bash
cd /home/dev62/Documents/agentic_rag
uv sync          # installs dependencies from pyproject.toml / uv.lock
```

#### Option 2: Using `pip`

Create and activate a virtual environment, then:

```bash
cd /home/dev62/Documents/agentic_rag
pip install -r requirements.txt
```

---

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

### API Endpoints (summary)

- **POST** `/ingest`
  - **Description**: Upload a PDF, extract text, chunk, embed, and store into Pinecone.
  - **Request**: `multipart/form-data` with field `file` (PDF only).
  - **Response**: JSON with message and number of chunks created.

- **POST** `/query`
  - **Description**: Ask a question; the agent retrieves context (via Pinecone or web search) and uses Groq to generate an answer.
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
      "answer": "Model-generated response..."
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

#### 2. Query the Agent

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who was Chhatrapati Shivaji?"
  }'
```

---

### Notes & Development

- The ingestion logic (loading, chunking, embedding) lives under `app/ingestion/`.
- Retrieval is implemented in `app/retriever/retriever.py` and used inside `rag_agent`.
- The agent uses **tool calls** (vector search + web search) via Groq; you can customize tools in `app/llm_tools/llm_tools.py`.
- For local experimentation, you can modify or extend `rag_agent` in `app/agent/rag_agent.py`.

