from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient

from rag.config import settings
from rag.core.chain import get_answer_llm
from rag.core.database import semantic_search, upsert_collection
from rag.core.ingest import get_embedding, get_embeddings, parse_file, text_splitter


def run_update_db(
    filepath: Path, qdrant_client: QdrantClient, openai_client: OpenAI
) -> None:
    """Parse file, split into chunks, embed and upload to Qdrant."""
    text = parse_file(filepath)
    chunks = text_splitter(text)
    embeddings = get_embeddings(openai_client, chunks)
    upsert_collection(qdrant_client, settings.COLLECTION_NAME, embeddings, chunks)


def run_rag_chain(
    user_query: str, qdrant_client: QdrantClient, openai_client: OpenAI
) -> str:
    """Sematic search and answer from LLM"""
    text_emb = get_embedding(openai_client, user_query)
    results = semantic_search(qdrant_client, text_emb)
    context = [str(text.payload["text"]) for text in results if text.payload]
    return get_answer_llm(openai_client, context, user_query)


if __name__ == "__main__":
    file_path = settings.RAW_DATA_DIR / "test.pdf"

    client_openai = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.OPENAI_API_KEY,
        max_retries=5,
    )

    client_qdrant = QdrantClient(url="http://localhost:6333")

    # run_update_db(file_path, client_qdrant, client_openai)
    print(run_rag_chain(client_qdrant, client_openai))
