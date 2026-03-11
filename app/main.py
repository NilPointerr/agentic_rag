import os

from fastapi import FastAPI

from app.api.routes import router
from app.vectorstore.pinecone_client import initialize_index

app = FastAPI(title="Agentic RAG API", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    """Initialize external dependencies during API startup."""

    if os.getenv("TESTING", "false").lower() == "true":
        return
    initialize_index()
