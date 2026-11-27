
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

from app.tools.catalog.repository import ToolCatalogRepository
from app.tools.catalog.service import ToolCatalogService
from app.tools.catalog.schemas import ToolServerCreate


router = APIRouter(
    prefix="/tool-servers",
    tags=["Tool Server Management"],
)


# ============================================================
# Dependency
# 每个请求获取独立的数据库 Session 和 Service 实例
# ============================================================

async def get_tool_catalog_service(
    db: AsyncSession = Depends(get_db),
) -> ToolCatalogService:

    repository = ToolCatalogRepository()

    return ToolCatalogService(
        repository=repository,
    )


# ============================================================
# 1. 注册 MCP Server
# POST /api/v1/tool-servers
# ============================================================

@router.post("")
async def register_tool_server(
    data: ToolServerCreate,
    service: ToolCatalogService = Depends(get_tool_catalog_service),
):
    try:
        result = await service.register_server(data)

        return {
            "success": True,
            "message": "Tool Server 注册成功",
            "data": jsonable_encoder(result),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


# ============================================================
# 2. 查询 Server 列表
# GET /api/v1/tool-servers
# ============================================================

@router.get("")
async def list_tool_servers(
    service: ToolCatalogService = Depends(get_tool_catalog_service),
):
    result = await service.list_servers()

    return {
        "success": True,
        "data": jsonable_encoder(result),
    }


# ============================================================
# 3. 查询 Server 详情
# GET /api/v1/tool-servers/{server_id}
# ============================================================

@router.get("/{server_id}")
async def get_tool_server(
    server_id: str,
    service: ToolCatalogService = Depends(get_tool_catalog_service),
):
    result = await service.get_server(server_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Tool Server 不存在",
        )

    return {
        "success": True,
        "data": jsonable_encoder(result),
    }


# ============================================================
# 4. 连接 Server 并同步工具
# POST /api/v1/tool-servers/{server_id}/connect
# ============================================================

@router.post("/{server_id}/connect")
async def connect_tool_server(
    server_id: str,
    service: ToolCatalogService = Depends(get_tool_catalog_service),
):
    try:
        result = await service.connect_server(server_id)

        return {
            "success": True,
            "message": "Tool Server 连接测试成功，工具同步完成",
            "data": jsonable_encoder(result),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Tool Server 连接或同步失败: {str(exc)}",
        )


# ============================================================
# 5. 手动重新同步工具
# POST /api/v1/tool-servers/{server_id}/sync
# ============================================================

@router.post("/{server_id}/sync")
async def sync_tool_server(
    server_id: str,
    service: ToolCatalogService = Depends(get_tool_catalog_service),
):
    try:
        result = await service.sync_server_tools(server_id)

        return {
            "success": True,
            "message": "工具同步完成",
            "data": jsonable_encoder(result),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"工具同步失败: {str(exc)}",
        )


# ============================================================
# 6. 断开 Server
# POST /api/v1/tool-servers/{server_id}/disconnect
# ============================================================

@router.post("/{server_id}/disconnect")
async def disconnect_tool_server(
    server_id: str,
    service: ToolCatalogService = Depends(get_tool_catalog_service),
):
    try:
        result = await service.disconnect_server(server_id)

        return {
            "success": bool(result),
            "message": "Tool Server 已断开",
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
