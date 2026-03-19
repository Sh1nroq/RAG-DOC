from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient

from rag.config import settings
from rag.core.database import upsert_collection
from rag.core.ingest import get_embeddings, parse_file, text_splitter


def run_pipeline(
    filepath: Path, qdrant_client: QdrantClient, openai_client: OpenAI
) -> None:
    text = parse_file(filepath)
    chunks = text_splitter(text)
    embeddings = get_embeddings(openai_client, chunks)
    upsert_collection(qdrant_client, settings.COLLECTION_NAME, embeddings, chunks)


if __name__ == "__main__":
    file_path = settings.RAW_DATA_DIR / "test.pdf"

    client_openai = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENAI_API_KEY,
        max_retries=5,
    )

    client_qdrant = QdrantClient(url="http://localhost:6333")

    run_pipeline(file_path, client_qdrant, client_openai)
