from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_and_store
from app.agent.rag_agent import rag_agent
from app.utils.logger import log_execution
from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import shutil
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_and_store
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

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/ingest")
@log_execution
async def ingest_documents(file: UploadFile = File(...)):

    try:
        # ✅ Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )

        # ✅ Save file locally
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ✅ Extract text from PDF
        text = load_pdf(file_path)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF"
            )

        # ✅ Chunk
        chunks = chunk_text(text)

        # ✅ Embed & store
        embed_and_store(chunks)

        return {
            "message": "✅ PDF ingested successfully",
            "chunks_created": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Query Endpoint
# ---------------------------

@router.post("/query")
@log_execution
def query_agent(request: QueryRequest):
    try:
        answer = rag_agent(request.query)
        return {
            "query": request.query,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))