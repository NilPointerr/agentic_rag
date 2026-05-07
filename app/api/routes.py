import os
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.agent.rag_agent import rag_agent
from app.config.settings import settings
from app.ingestion.chunker import chunk_pdf_pages
from app.ingestion.embedder import embed_and_store
from app.ingestion.pdf_loader import ImageOnlyPdfError, load_pdf_pages
from app.utils.logger import log_execution, logger
from app.vectorstore.pinecone_client import get_index_name

router = APIRouter()


# ---------------------------
# Request Models
# ---------------------------

class IngestRequest(BaseModel):
    directory: str = "data/sample_docs"


class QueryRequest(BaseModel):
    query: str


class SourceResponse(BaseModel):
    """Source chunk returned by internal hybrid retrieval or web search."""

    text: str | None = None
    title: str | None = None
    snippet: str | None = None
    score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    retrieval_score: float | None = None
    source_file: str | None = None
    source_path: str | None = None
    source_url: str | None = None
    page_number: int | None = None
    page_url: str | None = None
    chunk_index: int | None = None
    chunk_id: str | None = None
    source_type: str | None = None


class ImageResponse(BaseModel):
    """Image result returned when web fallback is used."""

    title: str | None = None
    image_url: str | None = None
    thumbnail_url: str | None = None
    source_url: str | None = None
    source: str | None = None


class QueryResponse(BaseModel):
    """Response payload for a RAG query."""

    query: str
    answer: str
    sources: list[SourceResponse]
    images: list[ImageResponse]


# ---------------------------
# Ingestion Endpoint
# ---------------------------

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/ingest")
@log_execution
async def ingest_documents(
    file: UploadFile = File(...),
    request: Request = None,
):
    """Save an uploaded PDF, extract its text, and store chunk embeddings."""
    if isinstance(file, Request) and hasattr(request, "filename"):
        file = request

    try:
        logger.info(f"Starting ingest for file: {file.filename}")
        filename = Path(file.filename or "upload.pdf").name

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            pages = load_pdf_pages(file_path)
        except ImageOnlyPdfError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        chunks = chunk_pdf_pages(
            pages=pages,
            source_file=filename,
            source_path=file_path,
            source_url=f"/uploads/{quote(filename)}",
        )

        logger.info(
            f"Created {len(chunks)} chunks for file '{filename}'. "
            f"Target Pinecone index: '{get_index_name()}'."
        )
        embed_and_store(chunks)

        return {
            "message": "✅ PDF ingested successfully",
            "file_name": filename,
            "chunks_created": len(chunks),
            "pinecone_index": get_index_name(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Query Endpoint
# ---------------------------

@router.post("/query", response_model=QueryResponse)
@log_execution
def query_agent(
    request: QueryRequest,
    http_request: Request = None,
) -> QueryResponse:
    """Run the RAG agent for a user query and return answer plus sources."""
    if isinstance(request, Request) and isinstance(http_request, QueryRequest):
        request = http_request

    try:
        if len(request.query) > settings.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Query exceeds the {settings.MAX_QUERY_LENGTH} character limit"
                ),
            )

        result = rag_agent(request.query)
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result.get("sources", []),
            images=result.get("images", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
