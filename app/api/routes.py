from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.agent.rag_agent import rag_agent
from app.config.settings import settings
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_and_store
from app.ingestion.pdf_loader import load_pdf_pages
from app.utils.logger import log_execution

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_rate_limit_state: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = Lock()


class QueryRequest(BaseModel):
    """Request payload for query endpoint."""

    query: str = Field(min_length=1, max_length=4000)


class QueryResponse(BaseModel):
    """Response payload for query endpoint."""

    query: str
    answer: str
    citations: list[dict]
    decision: str
    metrics: dict


def _assert_rate_limit(client_key: str) -> None:
    """Apply per-client fixed-window rate limit."""

    now = time.time()
    window_seconds = 60
    max_requests = settings.RATE_LIMIT_PER_MINUTE

    with _rate_limit_lock:
        entries = _rate_limit_state[client_key]
        while entries and now - entries[0] > window_seconds:
            entries.popleft()
        if len(entries) >= max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        entries.append(now)


def _validate_upload(file: UploadFile) -> str:
    """Validate uploaded file headers and return safe filename."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    safe_name = Path(file.filename).name
    if Path(safe_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content_type = (file.content_type or "").lower()
    if content_type and "pdf" not in content_type:
        raise HTTPException(status_code=400, detail="Invalid content type for PDF")

    return safe_name


async def _save_upload(
    file: UploadFile, destination: Path, max_size_bytes: int
) -> None:
    """Persist uploaded file in chunks with size enforcement and PDF magic check."""

    total_size = 0
    first_chunk = b""

    with destination.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break

            if not first_chunk:
                first_chunk = chunk

            total_size += len(chunk)
            if total_size > max_size_bytes:
                out.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413, detail="Uploaded file is too large"
                )

            out.write(chunk)

    if not first_chunk.startswith(b"%PDF"):
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid PDF signature")


def _build_chunk_payloads(
    pages: list[dict], source_name: str, doc_id: str
) -> list[dict]:
    """Convert page text into chunk payloads with document metadata."""

    payloads: list[dict] = []
    chunk_id = 0

    for page in pages:
        page_text = page.get("text", "")
        if not page_text:
            continue

        for chunk in chunk_text(page_text):
            payloads.append(
                {
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "page": page.get("page"),
                }
            )
            chunk_id += 1

    return payloads


@router.post("/ingest")
@log_execution
async def ingest_documents(file: UploadFile = File(...)) -> dict:
    """Upload a PDF, split to chunks, embed, and store in Pinecone."""

    try:
        safe_name = _validate_upload(file)
        destination = UPLOAD_DIR / safe_name
        max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        await _save_upload(file, destination, max_size)
        pages = load_pdf_pages(str(destination))
        doc_id = uuid4().hex
        chunks = _build_chunk_payloads(pages, safe_name, doc_id)

        if not chunks:
            raise HTTPException(
                status_code=400, detail="Could not extract text from PDF"
            )

        inserted = embed_and_store(chunks)
        return {
            "message": "PDF ingested successfully",
            "doc_id": doc_id,
            "chunks_created": len(chunks),
            "chunks_stored": inserted,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Ingestion failed") from exc


@router.post("/query", response_model=QueryResponse)
@log_execution
async def query_agent(request: QueryRequest, req: Request) -> QueryResponse:
    """Answer user query using retrieval-first agent with citations."""

    try:
        client_key = req.client.host if req.client else "unknown"
        _assert_rate_limit(client_key)

        result = await run_in_threadpool(rag_agent, request.query)
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            citations=result.get("citations", []),
            decision=result.get("decision", "unknown"),
            metrics=result.get("metrics", {}),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Query failed") from exc
