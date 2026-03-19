from numpy import ndarray
from qdrant_client import QdrantClient
from qdrant_client.grpc import VectorParams, Distance

from rag.config import settings


def upsert_collection(
    client: QdrantClient, collection_name: str, embeddings: ndarray, docs: str
) -> None:
    collection = client.get_collections().collections
    names_list = [names.name for names in collection]

    if collection_name not in names_list:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

    # for in range
    #
    # client.upsert(
    #     collection_name=collection_name,
    #     wait=True,
    #     points=[
    #         embeddings
    #     ],
    # )
