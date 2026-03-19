from numpy import ndarray
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

from rag.config import settings
from rag.core.ingest import chunk_to_id


def upsert_collection(
    client: QdrantClient, collection_name: str, embeddings: ndarray, chunks: str
) -> None:

    if not client.collection_exists(collection_name=settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

    client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[
            models.PointStruct(
                id=chunk_to_id(chunk), vector=embedding, payload={"text": chunk}
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ],
    )
