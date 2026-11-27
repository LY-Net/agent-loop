
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.model import bge_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时加载模型，关闭时释放模型。"""

    bge_model.load()

    try:
        yield
    finally:
        bge_model.unload()


app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.APP_DEBUG,
    lifespan=lifespan,
)


EmbeddingText = Annotated[
    str,
    Field(
        min_length=1,
        max_length=4000,
        description="单条文本长度不能超过 4000 个字符",
    ),
]


class EmbeddingRequest(BaseModel):
    texts: list[EmbeddingText] = Field(
        min_length=1,
        max_length=100,
        description="需要生成向量的文本列表，最多 100 条",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        """拒绝纯空白文本。"""

        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(
                    f"texts[{index}] cannot be empty or whitespace only"
                )

        return texts


class EmbeddingResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": settings.EMBEDDING_MODEL_NAME,
        "loaded": bge_model.is_loaded,
        "device": settings.EMBEDDING_DEVICE,
        "dimension": settings.EMBEDDING_DIMENSION,
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    try:
        vectors = bge_model.encode(request.texts)

        return EmbeddingResponse(
            model=settings.EMBEDDING_MODEL_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            embeddings=vectors,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Embedding inference failed",
        ) from exc
