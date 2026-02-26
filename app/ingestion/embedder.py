from sentence_transformers import SentenceTransformer
from uuid import uuid4
from app.utils import logger
from app.vectorstore.pinecone_client import index
from app.utils.logger import logger

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

def embed_and_store(chunks):
    embeddings = embed_texts(chunks)
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": f"doc-{uuid4()}",
            "values": embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)
    return embeddings

    
