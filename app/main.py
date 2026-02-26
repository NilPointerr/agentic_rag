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
from app.api.routes import router

app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0"
)

app.include_router(router)

# uv run uvicorn app.main:app --host 0.0.0.0 --port 8000