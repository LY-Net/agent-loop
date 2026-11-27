
from fastapi import HTTPException, Request, status

from app.embedding.service import EmbeddingService
from app.embedding.spaces.registry import EmbeddingSpaceRegistry
from app.embedding.spaces.service import EmbeddingSpaceService


def get_embedding_service(
    request: Request,
) -> EmbeddingService:

    embedding_service = getattr(
        request.app.state,
        "embedding_service",
        None,
    )

    if embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding Service is not initialized",
        )

    return embedding_service


def get_embedding_space_registry(
    request: Request,
) -> EmbeddingSpaceRegistry:

    space_registry = getattr(
        request.app.state,
        "embedding_space_registry",
        None,
    )

    if space_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding Space Registry is not initialized",
        )

    return space_registry


def get_embedding_space_service(
    request: Request,
) -> EmbeddingSpaceService:

    space_service = getattr(
        request.app.state,
        "embedding_space_service",
        None,
    )

    if space_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding Space Service is not initialized",
        )

    return space_service
