import os
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.agent.rag_agent import rag_agent
from app.config.settings import settings
from app.ingestion.chunker import chunk_pdf_pages
from app.ingestion.embedder import embed_and_store
from app.ingestion.pdf_loader import ImageOnlyPdfError, load_pdf_pages
from app.utils.logger import log_execution

router = APIRouter()


# ---------------------------
# Request Models
# ---------------------------

class IngestRequest(BaseModel):
    directory: str = "data/sample_docs"


class QueryRequest(BaseModel):
    query: str


# ---------------------------
# Ingestion Endpoint
# ---------------------------

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/ingest")
@log_execution
async def ingest_documents(file: UploadFile = File(...)):

    try:
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

        embed_and_store(chunks)

        return {
            "message": "✅ PDF ingested successfully",
            "file_name": filename,
            "chunks_created": len(chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Query Endpoint
# ---------------------------

@router.post("/query")
@log_execution
def query_agent(request: QueryRequest):
    try:
        if len(request.query) > settings.MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Query exceeds the {settings.MAX_QUERY_LENGTH} character limit"
                ),
            )

        result = rag_agent(request.query)
        return {
            "query": request.query,
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
