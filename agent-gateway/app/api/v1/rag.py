from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

from app.rag.service import RagService
from app.rag.hybrid import RagHybridSearchService
from app.rag.schemas import (
    RagDocumentIngestRequest,
    RagDocumentIngestResponse,
    RagDocumentDeleteResponse,
    RagSearchRequest,
    RagSearchResponse,
)


router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)


def get_rag_dependencies(request: Request):
    """获取应用生命周期中初始化的 RAG 依赖。"""

    embedding_service = getattr(
        request.app.state,
        "embedding_space_service",
        None,
    )

    space_registry = getattr(
        request.app.state,
        "embedding_space_registry",
        None,
    )

    if embedding_service is None or space_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG embedding service is unavailable",
        )

    return embedding_service, space_registry


@router.post(
    "/documents",
    response_model=RagDocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建或更新 RAG 文档",
)
async def ingest_document(
    payload: RagDocumentIngestRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
):
    embedding_service, space_registry = get_rag_dependencies(request)

    service = RagService(
        session=session,
        embedding_space_service=embedding_service,
        space_registry=space_registry,
    )

    try:
        return await service.ingest(payload)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.delete(
    "/documents/{document_id}",
    response_model=RagDocumentDeleteResponse,
    summary="删除 RAG 文档",
)
async def delete_document(
    document_id: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
):
    embedding_service, space_registry = get_rag_dependencies(request)

    service = RagService(
        session=session,
        embedding_space_service=embedding_service,
        space_registry=space_registry,
    )

    return await service.delete(document_id)


@router.post("/search", response_model=RagSearchResponse)
async def search_documents(
    payload: RagSearchRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
):
    service = RagHybridSearchService(
        session=session,
        embedding_space_service=request.app.state.embedding_space_service,
        space_registry=request.app.state.embedding_space_registry,
    )

    return await service.search(payload)