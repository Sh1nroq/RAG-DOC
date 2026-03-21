from fastapi import APIRouter, HTTPException

from rag.backend.repository import rag_repo
from rag.backend.schemas import ModelResponse, UserRequest, StatusResponse

router = APIRouter(
    prefix="/app",
)


@router.post("/search")
async def search(user_request: UserRequest) -> ModelResponse:
    model_query = await rag_repo.semantic_search(user_request)
    if not model_query.model_response:
        raise HTTPException(status_code=500, detail="Проблемы с ответом ИИ")

    return model_query


@router.post("/upload")
async def upload(user_request: UserRequest) -> StatusResponse:
    upload_query = await rag_repo.upload_document(user_request)
    return upload_query
