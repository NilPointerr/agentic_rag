import asyncio
import os
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.agent.rag_agent import rag_agent
from app.config.settings import settings
from app.ingestion.chunker import chunk_pdf_pages
from app.ingestion.dedup_store import (
    ChunkDedupRecord,
    DeduplicationError,
    DeduplicationStore,
    DocumentDedupRecord,
    utc_now_iso,
)
from app.ingestion.embedder import embed_and_store
from app.ingestion.hash_utils import generate_content_hash, normalize_text
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


class IngestResponse(BaseModel):
    """Structured response describing what happened during PDF ingestion.

    `chunks_created` counts the chunks produced by the chunker, while
    `chunks_embedded` counts only the chunks that were not skipped by exact hash
    deduplication.
    """

    message: str
    file_name: str
    chunks_created: int
    chunks_embedded: int
    chunks_skipped_duplicate: int
    pinecone_index: str
    skipped_duplicate: bool
    duplicate_reason: str | None
    document_hash: str | None

    def __getitem__(self, key: str):
        """Keep direct route tests compatible with dict-style access."""
        return getattr(self, key)


# ---------------------------
# Ingestion Endpoint
# ---------------------------

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_upload_file(file: UploadFile, file_path: str) -> None:
    """Write the uploaded PDF to disk from a worker thread."""
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


@router.post("/ingest", response_model=IngestResponse)
@log_execution
async def ingest_documents(
    file: UploadFile = File(...),
    request: Request = None,
) -> IngestResponse:
    """Ingest a PDF into the retrieval stores with exact hash deduplication.

    Workflow:
    1. Save the uploaded file locally.
    2. Extract text from the PDF.
    3. Hash the normalized full-document text and skip ingest if it already
       exists.
    4. Chunk the document and hash each chunk.
    5. Reserve only unseen chunk hashes, then embed and index those chunks.
    6. Persist final document and chunk status in the dedup registry.
    """
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

        await asyncio.to_thread(_save_upload_file, file, file_path)

        try:
            pages = await asyncio.to_thread(load_pdf_pages, file_path)
        except ImageOnlyPdfError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        document_text = normalize_text(
            " ".join(str(page.get("text", "")) for page in pages)
        )
        document_hash = generate_content_hash(document_text)
        uploaded_at = utc_now_iso()
        dedup_store = DeduplicationStore()

        document_record = DocumentDedupRecord(
            document_hash=document_hash,
            uploaded_at=uploaded_at,
            source_filename=filename,
            embedding_model=settings.EMBEDDING_MODEL,
            status="processing",
        )
        should_process_document = await asyncio.to_thread(
            dedup_store.register_document_if_absent,
            document_record,
        )

        if not should_process_document:
            logger.info(
                f"Skipping duplicate document ingest for '{filename}'. "
                f"document_hash={document_hash}"
            )
            return IngestResponse(
                message="Duplicate PDF skipped",
                file_name=filename,
                chunks_created=0,
                chunks_embedded=0,
                chunks_skipped_duplicate=0,
                pinecone_index=get_index_name(),
                skipped_duplicate=True,
                duplicate_reason="document_hash_exists",
                document_hash=document_hash,
            )

        chunks = await asyncio.to_thread(
            chunk_pdf_pages,
            pages=pages,
            source_file=filename,
            source_path=file_path,
            source_url=f"/uploads/{quote(filename)}",
        )

        for chunk in chunks:
            chunk_hash = generate_content_hash(chunk["text"])
            chunk["document_hash"] = document_hash
            chunk["chunk_hash"] = chunk_hash
            chunk["chunk_id"] = chunk_hash
            chunk["uploaded_at"] = uploaded_at
            chunk["source_filename"] = filename
            chunk["embedding_model"] = settings.EMBEDDING_MODEL

        chunk_records = [
            ChunkDedupRecord(
                chunk_hash=chunk["chunk_hash"],
                document_hash=document_hash,
                uploaded_at=uploaded_at,
                source_filename=filename,
                embedding_model=settings.EMBEDDING_MODEL,
                status="processing",
            )
            for chunk in chunks
        ]
        newly_reserved_chunk_hashes = await asyncio.to_thread(
            dedup_store.reserve_chunks,
            chunk_records,
        )
        chunks_to_embed = [
            chunk
            for chunk in chunks
            if chunk["chunk_hash"] in newly_reserved_chunk_hashes
        ]
        skipped_chunk_count = len(chunks) - len(chunks_to_embed)

        logger.info(
            f"Created {len(chunks)} chunks for file '{filename}'. "
            f"Embedding {len(chunks_to_embed)} new chunk(s), "
            f"skipping {skipped_chunk_count} duplicate chunk(s). "
            f"Target Pinecone index: '{get_index_name()}'."
        )

        if chunks_to_embed:
            try:
                await asyncio.to_thread(embed_and_store, chunks_to_embed)
            except Exception:
                await asyncio.to_thread(
                    dedup_store.mark_chunks_status,
                    [chunk["chunk_hash"] for chunk in chunks_to_embed],
                    "failed",
                )
                await asyncio.to_thread(
                    dedup_store.upsert_document,
                    DocumentDedupRecord(
                        document_hash=document_hash,
                        uploaded_at=uploaded_at,
                        source_filename=filename,
                        embedding_model=settings.EMBEDDING_MODEL,
                        status="failed",
                    ),
                )
                raise

            indexed_chunk_records = [
                ChunkDedupRecord(
                    chunk_hash=chunk["chunk_hash"],
                    document_hash=document_hash,
                    uploaded_at=uploaded_at,
                    source_filename=filename,
                    embedding_model=settings.EMBEDDING_MODEL,
                    status="indexed",
                )
                for chunk in chunks_to_embed
            ]
            await asyncio.to_thread(dedup_store.register_chunks, indexed_chunk_records)

        await asyncio.to_thread(
            dedup_store.upsert_document,
            DocumentDedupRecord(
                document_hash=document_hash,
                uploaded_at=uploaded_at,
                source_filename=filename,
                embedding_model=settings.EMBEDDING_MODEL,
                status="indexed",
            ),
        )

        return IngestResponse(
            message="✅ PDF ingested successfully",
            file_name=filename,
            chunks_created=len(chunks),
            chunks_embedded=len(chunks_to_embed),
            chunks_skipped_duplicate=skipped_chunk_count,
            pinecone_index=get_index_name(),
            skipped_duplicate=bool(chunks and not chunks_to_embed),
            duplicate_reason=(
                "all_chunks_duplicate" if chunks and not chunks_to_embed else None
            ),
            document_hash=document_hash,
        )

    except HTTPException:
        raise
    except DeduplicationError as e:
        logger.error(f"Deduplication failed for file '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
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
