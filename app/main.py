# from app.ingestion.loader import load_documents
# from app.ingestion.chunker import chunk_text
# from app.ingestion.embedder import embed_and_store

# def ingest():
#     docs = load_documents("data/sample_docs")
    
#     all_chunks = []
#     for doc in docs:
#         chunks = chunk_text(doc)
#         all_chunks.extend(chunks)

#     embed_and_store(all_chunks)
#     print("✅ Documents ingested successfully!")

# if __name__ == "__main__":
#     ingest()


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config.settings import settings

app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
