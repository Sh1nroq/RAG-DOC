import asyncio

from openai import OpenAI
from qdrant_client import QdrantClient

from rag.backend.schemas import ModelResponse, StatusResponse, UserRequest
from rag.config import settings
from rag.core.pipeline import run_rag_chain, run_update_db


class RagRepository:
    def __init__(self) -> None:
        self.client_openai = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.OPENAI_API_KEY,
            max_retries=5,
        )

        self.client_qdrant = QdrantClient(url="http://localhost:6333")

    async def upload_document(self, request: UserRequest) -> StatusResponse:
        filepath = settings.RAW_DATA_DIR / request.doc_name
        await asyncio.to_thread(
            run_update_db, filepath, self.client_qdrant, self.client_openai
        )
        return StatusResponse(
            status="success", message=f"Файл {request.doc_name} успешно загружен!"
        )

    async def semantic_search(self, request: UserRequest) -> ModelResponse:
        result = await asyncio.to_thread(
            run_rag_chain, request.user_request, self.client_qdrant, self.client_openai
        )
        return ModelResponse(model_response=result)


rag_repo = RagRepository()
