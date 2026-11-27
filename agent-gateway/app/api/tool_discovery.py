from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

from app.embedding.dependencies import get_embedding_space_service
from app.embedding.spaces.service import EmbeddingSpaceService

from app.tools.catalog.schemas import ToolDiscoveryQuery
from app.tools.discovery.repository import ToolDiscoveryRepository
from app.tools.discovery.service import ToolDiscoveryService


router = APIRouter(
    prefix="/tool-discovery",
    tags=["Tool Discovery"],
)


def get_tool_discovery_service(
    db: AsyncSession = Depends(get_db),
    embedding_space_service: EmbeddingSpaceService = Depends(
        get_embedding_space_service
    ),
) -> ToolDiscoveryService:

    repository = ToolDiscoveryRepository(
        session=db,
    )

    return ToolDiscoveryService(
        repository=repository,
        embedding_space_service=embedding_space_service,
    )


@router.post("/search")
async def discover_tools(
    query: ToolDiscoveryQuery,
    service: ToolDiscoveryService = Depends(
        get_tool_discovery_service
    ),
):
    try:
        result = await service.discover(query)
        return {
            "success": True,
            "data": result.model_dump(mode="json"),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="工具发现失败",
        )