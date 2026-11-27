
"""
Tool Catalog Service

负责：
1. Tool Server 管理。
2. MCP 工具发现与同步。
3. 工具目录查询和 Schema 加载。
4. Tool Catalog 与 Tool Embedding 事务协调。
"""

import math
from typing import Any

from app.embedding.schemas import EmbeddingInput
from app.embedding.spaces.service import EmbeddingSpaceService

from app.tools.catalog.schemas import (
    ToolProviderType,
    ToolServerCreate,
    ToolServerInfo,
    CatalogToolCreate,
    CatalogToolSummary,
    CatalogToolDetail,
    ToolDiscoveryQuery,
    ToolServerStatus,
    CatalogToolStatus,
)

from app.tools.providers.mcp import mcp_provider
from app.tools.catalog.repository import ToolCatalogRepository
from app.tools.catalog.embedding_repository import (
    ToolEmbeddingRepository,
)


TOOL_DISCOVERY_SPACE_ID = "tool_discovery"


class ToolCatalogService:

    def __init__(
        self,
        repository: ToolCatalogRepository,
        embedding_space_service: EmbeddingSpaceService,
        provider=None,
    ):
        self.repository = repository
        self.embedding_space_service = embedding_space_service
        self.provider = provider or mcp_provider

        self.session = repository.session

        self.embedding_repository = ToolEmbeddingRepository(
            session=self.session
        )

    # =========================================================
    # 一、Server 管理
    # =========================================================

    async def register_server(
        self,
        data: ToolServerCreate,
    ) -> ToolServerInfo:

        provider_type = self._normalize_provider_type(
            data.provider_type
        )

        if provider_type == ToolProviderType.BUILTIN:
            raise ValueError(
                "BUILTIN 类型不需要注册 Tool Server"
            )

        if not data.server_url:
            raise ValueError(
                f"{provider_type.value} 类型必须提供 server_url"
            )

        async with self.session.begin():
            existing = await self.repository.get_server_by_name(
                data.name
            )

            if existing is not None:
                raise ValueError(
                    f"Tool Server 名称已经存在: {data.name}"
                )

            return await self.repository.create_server(data)

    async def get_server(
        self,
        server_id: str,
    ) -> ToolServerInfo | None:

        if not server_id:
            raise ValueError("server_id 不能为空")

        return await self.repository.get_server(server_id)

    async def list_servers(self) -> list[ToolServerInfo]:
        return await self.repository.list_servers()

    async def update_server_status(
        self,
        server_id: str,
        status: str,
    ) -> bool:

        allowed_statuses = {
            "pending",
            "connected",
            "disconnected",
            "error",
            "disabled",
        }

        if status not in allowed_statuses:
            raise ValueError(
                f"不支持的 Server 状态: {status}"
            )

        async with self.session.begin():
            return await self.repository.update_server_status(
                server_id=server_id,
                status=status,
            )

    # =========================================================
    # 二、工具发现与查询
    # =========================================================

    async def discover_tools(
        self,
        query: ToolDiscoveryQuery,
    ) -> list[CatalogToolSummary]:

        if query.top_k < 1 or query.top_k > 100:
            raise ValueError(
                "top_k 必须在 1 到 100 之间"
            )

        return await self.repository.list_tool_summaries(
            query
        )

    async def load_tool(
        self,
        tool_id: str,
    ) -> CatalogToolDetail | None:

        if not tool_id:
            raise ValueError("tool_id 不能为空")

        tool = await self.repository.get_tool_detail(tool_id)

        if tool is None or tool.status != "active":
            return None

        return tool

    async def get_tool_detail(
        self,
        tool_id: str,
    ) -> CatalogToolDetail | None:

        if not tool_id:
            raise ValueError("tool_id 不能为空")

        return await self.repository.get_tool_detail(tool_id)

    # =========================================================
    # 三、工具状态管理
    # =========================================================

    async def update_tool_status(
        self,
        tool_id: str,
        status: str,
    ) -> bool:

        allowed_statuses = {
            "active",
            "disabled",
            "unavailable",
            "deleted",
        }

        if status not in allowed_statuses:
            raise ValueError(
                f"不支持的工具状态: {status}"
            )

        async with self.session.begin():
            updated = await self.repository.update_tool_status(
                tool_id=tool_id,
                status=status,
            )

            if updated and status != "active":
                await self.embedding_repository.delete_by_tool_ids(
                    [tool_id],
                    TOOL_DISCOVERY_SPACE_ID,
                )

            return updated

    # =========================================================
    # 四、MCP 连接
    # =========================================================

    async def connect_server(
        self,
        server_id: str,
    ) -> dict[str, Any]:

        try:
            sync_result = await self.sync_server_tools(server_id)

            server = await self.repository.get_server(server_id)

            if server is None:
                await self.session.rollback()
                raise ValueError(
                    f"Tool Server 不存在: {server_id}"
                )

            result = {
                "success": True,
                "server_id": server_id,
                "server_name": server.name,
                "provider_type": server.provider_type,
                "sync_result": sync_result,
            }

            await self.session.rollback()

            return result

        except Exception:
            if self.session.in_transaction():
                await self.session.rollback()

            async with self.session.begin():
                await self.repository.update_server_status(
                    server_id,
                    ToolServerStatus.ERROR,
                )

            raise

    async def disconnect_server(
        self,
        server_id: str,
    ) -> bool:

        async with self.session.begin():
            server = await self.repository.get_server(server_id)

            if server is None:
                raise ValueError(
                    f"Tool Server 不存在: {server_id}"
                )

            return await self.repository.update_server_status(
                server_id,
                ToolServerStatus.DISCONNECTED,
            )

    # =========================================================
    # 五、MCP 工具同步 + Embedding
    # =========================================================

    async def sync_server_tools(
        self,
        server_id: str,
    ) -> dict[str, Any]:
        """
        完整同步流程：

        1. 读取 Server 配置。
        2. 结束只读数据库事务。
        3. 通过 MCP 获取工具列表。
        4. 构造工具语义文本。
        5. 通过 tool_discovery Space 生成向量。
        6. 开启数据库事务。
        7. 同步 Tool Catalog。
        8. Upsert Tool Embedding。
        9. 清理非 active 工具的向量。
        10. 原子提交。
        """

        # -----------------------------------------------------
        # 1. 读取 Server 配置
        # -----------------------------------------------------

        server = await self.repository.get_server(server_id)

        if server is None:
            raise ValueError(
                f"Tool Server 不存在: {server_id}"
            )

        provider_type = self._normalize_provider_type(
            server.provider_type
        )

        if provider_type != ToolProviderType.MCP:
            raise ValueError(
                f"当前不支持该 Provider 的工具同步: {provider_type}"
            )

        if not server.server_url:
            raise ValueError("MCP Server URL 不能为空")

        server_url = server.server_url
        connection_config = server.connection_config or {}

        # 结束只读事务，避免网络调用期间占用数据库事务
        await self.session.rollback()

        # -----------------------------------------------------
        # 2. MCP 工具发现
        # -----------------------------------------------------

        transport = connection_config.get(
            "transport",
            "streamable_http",
        )

        discovered_tools = await self.provider.list_tools(
            server_url,
            transport=transport,
        )

        catalog_tools: list[CatalogToolCreate] = []

        for tool in discovered_tools:

            tool_name = tool.get("name")

            if not tool_name:
                continue

            catalog_tools.append(
                CatalogToolCreate(
                    server_id=server_id,
                    provider_type=ToolProviderType.MCP,
                    name=tool_name,
                    description=tool.get("description", ""),
                    capabilities=tool.get("capabilities", []),
                    keywords=tool.get("keywords", []),
                    input_schema=tool.get("input_schema", {}),
                    metadata=tool.get("metadata", {}),
                    status=CatalogToolStatus.ACTIVE,
                )
            )

        # 防止同一 Server 返回重复工具名称
        tool_names = [item.name for item in catalog_tools]

        if len(tool_names) != len(set(tool_names)):
            raise ValueError(
                "MCP Server 返回了重复的工具名称"
            )

        # -----------------------------------------------------
        # 3. 构造工具语义文本
        # -----------------------------------------------------

        embedding_texts: list[str] = []

        for tool in catalog_tools:
            embedding_texts.append(
                self._build_embedding_text(tool)
            )

        # -----------------------------------------------------
        # 4. 调用 Embedding Service
        # -----------------------------------------------------

        vectors: list[list[float]] = []

        if embedding_texts:
            vectors = await self.embedding_space_service.embed(
                space_id=TOOL_DISCOVERY_SPACE_ID,
                request=EmbeddingInput(
                    texts=embedding_texts,
                ),
            )

        space = self.embedding_space_service.get_space(
            TOOL_DISCOVERY_SPACE_ID
        )

        model_info = self.embedding_space_service.model_info

        if len(vectors) != len(catalog_tools):
            raise RuntimeError(
                "Embedding 返回数量与工具数量不一致"
            )

        for index, vector in enumerate(vectors):

            if len(vector) != space.dimension:
                raise RuntimeError(
                    f"工具 {catalog_tools[index].name} "
                    f"Embedding 维度错误"
                )

            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError(
                    f"工具 {catalog_tools[index].name} "
                    f"Embedding 包含非有限值"
                )

        # -----------------------------------------------------
        # 5. 建立名称到向量的映射
        # -----------------------------------------------------

        vector_map = {
            tool.name: {
                "embedding_text": embedding_texts[index],
                "embedding": vectors[index],
            }
            for index, tool in enumerate(catalog_tools)
        }

        # -----------------------------------------------------
        # 6. 开启数据库事务，原子同步
        # -----------------------------------------------------

        async with self.session.begin():

            sync_result = await self.repository.sync_server_tools(
                server_id,
                catalog_tools,
            )

            summaries: list[CatalogToolSummary] = (
                sync_result["tools"]
            )

            embedding_rows = []
            inactive_tool_ids = []

            for summary in summaries:

                if summary.status != "active":
                    inactive_tool_ids.append(summary.tool_id)
                    continue

                vector_data = vector_map.get(summary.name)

                if vector_data is None:
                    continue

                embedding_rows.append({
                    "tool_id": summary.tool_id,
                    "space_id": TOOL_DISCOVERY_SPACE_ID,
                    "embedding_text": vector_data["embedding_text"],
                    "embedding": vector_data["embedding"],
                    "model_fingerprint": model_info.fingerprint,
                })

            # 新增或更新 active 工具向量
            embedding_count = (
                await self.embedding_repository.upsert_many(
                    embedding_rows
                )
            )

            # 删除 disabled / unavailable / deleted 工具向量
            deleted_embedding_count = (
                await self.embedding_repository.delete_by_tool_ids(
                    inactive_tool_ids,
                    TOOL_DISCOVERY_SPACE_ID,
                )
            )

            sync_result["embedding_count"] = embedding_count
            sync_result["deleted_embedding_count"] = (
                deleted_embedding_count
            )

            sync_result.pop("tools", None)

            return sync_result

    # =========================================================
    # 六、内部方法
    # =========================================================

    @staticmethod
    def _build_embedding_text(
        tool: CatalogToolCreate,
    ) -> str:
        """
        构造工具语义检索文本。

        不把完整 JSON Schema 放进 Embedding 文本，
        避免参数定义噪声影响工具语义匹配。
        """

        capabilities = "、".join(tool.capabilities or [])
        keywords = "、".join(tool.keywords or [])

        return (
            f"工具名称：{tool.name}\n"
            f"功能描述：{tool.description}\n"
            f"能力标签：{capabilities}\n"
            f"关键词：{keywords}"
        )

    @staticmethod
    def _normalize_provider_type(
        provider_type: ToolProviderType | str,
    ) -> ToolProviderType:

        if isinstance(provider_type, ToolProviderType):
            return provider_type

        try:
            return ToolProviderType(provider_type)
        except ValueError:
            raise ValueError(
                f"不支持的 Provider 类型: {provider_type}"
            )
