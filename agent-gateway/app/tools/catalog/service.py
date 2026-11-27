
"""
Tool Catalog Service

工具目录业务服务层。

职责：
1. 管理 Tool Server 注册。
2. 查询 Server 配置。
3. 查询工具目录。
4. 根据需求发现候选工具。
5. 加载选中工具的完整定义。
6. 编排 MCP Server 工具同步。

分层关系：

API / Agent
    |
    v
ToolCatalogService
    |
    v
ToolCatalogRepository
    |
    v
PostgreSQL

注意：
本模块暂时不负责 MCP 网络连接。
MCP Client 后续会调用本服务完成工具目录同步。
"""

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

from typing import Any

from app.tools.providers.mcp import mcp_provider
from app.tools.catalog.repository import ToolCatalogRepository

class ToolCatalogService:
    """
    Tool Catalog 业务服务。

    Repository 负责数据库操作；
    Service 负责业务规则和流程编排。
    """

    def __init__(
            self,
            repository: ToolCatalogRepository,
            provider=None,
    ):
        self.repository = repository
        self.provider = provider or mcp_provider

    # =========================================================
    # 一、Server 管理
    # =========================================================

    async def register_server(
        self,
        data: ToolServerCreate,
    ) -> ToolServerInfo:
        """
        注册一个外部工具服务。

        例如：

        {
            "name": "sales-mcp",
            "provider_type": "mcp",
            "server_url": "http://127.0.0.1:9000/mcp"
        }

        当前阶段只保存连接配置，不真正建立网络连接。

        后续 MCP Client 会负责：
        - 建立连接
        - 获取工具列表
        - 更新 Server 状态
        - 同步工具目录
        """

        provider_type = self._normalize_provider_type(
            data.provider_type
        )

        # 当前 Server 表用于管理外部服务。
        # 系统内置工具直接进入工具目录，不需要注册 Server。
        if provider_type == ToolProviderType.BUILTIN:
            raise ValueError(
                "BUILTIN 类型不需要注册 Tool Server，"
                "内置工具应直接注册到 Tool Catalog"
            )

        # MCP 和 Remote 服务必须提供连接地址
        if not data.server_url:
            raise ValueError(
                f"{provider_type.value} 类型必须提供 server_url"
            )

        # 创建前检查名称是否已存在
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
        """查询指定 Server。"""

        if not server_id:
            raise ValueError("server_id 不能为空")

        return await self.repository.get_server(server_id)

    async def list_servers(
        self,
    ) -> list[ToolServerInfo]:
        """获取全部 Server 配置。"""

        return await self.repository.list_servers()

    async def update_server_status(
        self,
        server_id: str,
        status: str,
    ) -> bool:
        """更新 Server 连接状态。"""

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

        return await self.repository.update_server_status(
            server_id=server_id,
            status=status,
        )

    # =========================================================
    # 二、工具发现
    # =========================================================

    async def discover_tools(
        self,
        query: ToolDiscoveryQuery,
    ) -> list[CatalogToolSummary]:
        """
        根据查询条件发现候选工具。

        这里只返回轻量工具信息，不加载完整 Schema。

        查询条件可以包括：
        - query：自然语言关键词
        - capabilities：要求具备的能力
        - keywords：关键词列表
        - provider_types：工具来源类型
        - top_k：最多返回多少个候选工具

        后续可以在这里接入：
        - 关键词检索
        - 能力标签匹配
        - 语义向量检索
        - 用户权限过滤
        - 候选工具相关性排序
        """

        if query.top_k < 1 or query.top_k > 100:
            raise ValueError(
                "top_k 必须在 1 到 100 之间"
            )

        return await self.repository.list_tool_summaries(
            query
        )

    # =========================================================
    # 三、加载工具完整定义
    # =========================================================

    async def load_tool(
        self,
        tool_id: str,
    ) -> CatalogToolDetail | None:
        """
        加载单个工具的完整定义。

        Discovery 阶段只返回轻量信息。
        只有候选工具被选中后，才调用此方法。

        这样可以避免把所有工具的 JSON Schema
        一次性加载到 Agent 上下文中。
        """

        if not tool_id:
            raise ValueError("tool_id 不能为空")

        tool = await self.repository.get_tool_detail(
            tool_id
        )

        if tool is None:
            return None

        # 不允许加载不可用或已禁用的工具用于执行
        if tool.status != "active":
            return None

        return tool

    async def get_tool_detail(
        self,
        tool_id: str,
    ) -> CatalogToolDetail | None:
        """
        管理用途：查询工具完整信息。

        与 load_tool 不同：
        这里允许查看 disabled / unavailable 工具，
        方便管理后台排查工具状态。
        """

        if not tool_id:
            raise ValueError("tool_id 不能为空")

        return await self.repository.get_tool_detail(
            tool_id
        )



    # =========================================================
    # 五、工具状态管理
    # =========================================================

    async def update_tool_status(
        self,
        tool_id: str,
        status: str,
    ) -> bool:
        """更新工具状态。"""

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

        return await self.repository.update_tool_status(
            tool_id=tool_id,
            status=status,
        )

    # =========================================================
    # 六、内部辅助方法
    # =========================================================

    @staticmethod
    def _normalize_provider_type(
        provider_type: ToolProviderType | str,
    ) -> ToolProviderType:
        """
        统一 Provider 类型。

        兼容：
        - ToolProviderType.MCP
        - "mcp"
        """

        if isinstance(provider_type, ToolProviderType):
            return provider_type

        try:
            return ToolProviderType(provider_type)
        except ValueError:
            raise ValueError(
                f"不支持的 Provider 类型: {provider_type}"
            )

    async def connect_server(self, server_id: str) -> dict[str, Any]:
        """
        根据数据库中的 Server 配置建立一次 MCP 连接探测，
        自动发现工具，并将工具同步到 Tool Catalog。

        注意：
        当前 MCP Provider 使用按需连接模式。
        本方法完成后不会保留长期连接。
        """

        # 1. 从数据库读取 Server 配置
        server = await self.repository.get_server(server_id)

        if server is None:
            raise ValueError(f"Tool Server 不存在: {server_id}")

        # 2. 当前版本先只支持 MCP Provider
        provider_type = self._normalize_provider_type(
            server.provider_type
        )

        if provider_type != ToolProviderType.MCP:
            raise ValueError(
                f"当前暂不支持连接该 Provider: {provider_type}"
            )

        if not server.server_url:
            raise ValueError("MCP Server URL 不能为空")

        try:
            # 3. 标记为连接中
            await self.repository.update_server_status(
                server_id,
                ToolServerStatus.PENDING,
            )

            # 4. 自动发现并同步工具
            sync_result = await self.sync_server_tools(server_id)

            # 5. 同步成功，更新状态
            await self.repository.update_server_status(
                server_id,
                ToolServerStatus.CONNECTED,
            )

            return {
                "success": True,
                "server_id": server_id,
                "server_name": server.name,
                "provider_type": provider_type,
                "sync_result": sync_result,
            }

        except Exception:
            # 连接或工具同步失败
            await self.repository.update_server_status(
                server_id,
                ToolServerStatus.ERROR,
            )
            raise

    async def sync_server_tools(
            self,
            server_id: str,
    ) -> dict[str, int]:
        """
        根据 Server ID 重新发现工具，并同步到 Tool Catalog。

        不接收外部传入的 URL 或工具列表。
        URL 和工具信息均由数据库配置及 MCP Server 自动获取。
        """

        # 1. 从数据库获取 Server
        server = await self.repository.get_server(server_id)

        if server is None:
            raise ValueError(f"Tool Server 不存在: {server_id}")

        provider_type = self._normalize_provider_type(
            server.provider_type
        )

        if provider_type != ToolProviderType.MCP:
            raise ValueError(
                f"当前不支持该 Provider 的工具同步: {provider_type}"
            )

        if not server.server_url:
            raise ValueError("MCP Server URL 不能为空")

        # 2. 调用 MCP Provider 获取远端工具
        # Provider 内部负责建立会话、调用 tools/list、关闭会话
        transport = (
                server.connection_config or {}
        ).get("transport", "streamable_http")

        print(
            f"[MCP DEBUG] server_id={server_id}, "
            f"transport={transport}, "
            f"url={server.server_url}"
        )

        discovered_tools = await self.provider.list_tools(
            server.server_url,
            transport=transport,
        )

        # 3. 将 MCP 工具转换成统一 Catalog 数据结构
        catalog_tools = []

        for tool in discovered_tools:

            tool_name = tool.get("name")

            if not tool_name:
                continue

            catalog_tool = CatalogToolCreate(
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

            catalog_tools.append(catalog_tool)

        # 4. 将发现结果交给 Repository 统一同步
        # Repository 负责新增、更新及标记已经消失的工具
        return await self.repository.sync_server_tools(
            server_id,
            catalog_tools,
        )

    async def disconnect_server(self, server_id: str) -> bool:
        """
        断开 Server。

        当前 Provider 是按需连接模式，没有长期持有的会话，
        因此这里主要更新 Catalog 中的连接状态。

        后续实现 Connection Manager 后，再增加真实会话释放逻辑。
        """

        server = await self.repository.get_server(server_id)

        if server is None:
            raise ValueError(f"Tool Server 不存在: {server_id}")

        return await self.repository.update_server_status(
            server_id,
            ToolServerStatus.DISCONNECTED,
        )



# 全局 Service 实例
# tool_catalog_service = ToolCatalogService()
