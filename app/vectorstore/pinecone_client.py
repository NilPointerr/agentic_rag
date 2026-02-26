from pinecone import Pinecone, ServerlessSpec
from app.config.settings import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)

index_name = settings.PINECONE_INDEX_NAME

# Check if index exists
existing_indexes = [i["name"] for i in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=settings.EMBEDDING_DIMENSION,  # 384
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)