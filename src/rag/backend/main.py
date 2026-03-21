from fastapi import FastAPI

from rag.backend.router import router

app = FastAPI()
app.include_router(router)
