from pydantic import BaseModel


class UserRequest(BaseModel):
    doc_name: str
    user_request: str


class ModelResponse(BaseModel):
    model_response: str


class StatusResponse(BaseModel):
    status: str
    message: str
