
"""
Tool Catalog Repository

职责：
1. 管理 Tool Server 配置。
2. 管理工具目录元数据。
3. 提供轻量工具列表查询。
4. 按需加载完整工具 Schema。
5. 支持 MCP Server 工具列表的批量同步。

设计原则：
- Repository 只负责数据库操作，不负责网络请求。
- 每个公开方法使用独立 AsyncSession。
- 批量同步在一个事务中完成。
- 不物理删除同步失效的工具，而是标记为 unavailable。
"""

from typing import Any

from sqlalchemy import select, update, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal

from app.models.tool_catalog import ToolServer, CatalogTool

from app.tools.catalog.schemas import (
    ToolServerCreate,
    ToolServerInfo,
    CatalogToolCreate,
    CatalogToolSummary,
    CatalogToolDetail,
    ToolDiscoveryQuery,
)


class ToolCatalogRepository:
    """
    Tool Catalog 数据访问层。

    这里不处理 MCP 协议，也不执行工具。
    所有方法只负责 PostgreSQL 数据读写。
    """

    # =========================================================
    # 一、Tool Server 管理
    # =========================================================

    async def create_server(
        self,
        data: ToolServerCreate,
    ) -> ToolServerInfo:
        """
        创建一个工具服务配置。

        例如：
        - 新增一个 MCP Server
        - 新增一个 Remote Tool Server

        authentication_ref 只保存凭证引用，不保存真实 Token。
        """

        async with AsyncSessionLocal() as session:
            server = ToolServer(
                name=data.name,
                provider_type=data.provider_type.value
                if hasattr(data.provider_type, "value")
                else data.provider_type,
                server_url=str(data.server_url) if data.server_url else None,
                description=data.description,
                authentication_ref=data.authentication_ref,
                connection_config=data.connection_config or {},
            )

            session.add(server)
            await session.commit()
            await session.refresh(server)

            return self._to_server_info(server)

    async def get_server(
        self,
        server_id: str,
    ) -> ToolServerInfo | None:
        """根据 server_id 查询 Server 配置。"""

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ToolServer).where(
                    ToolServer.server_id == server_id
                )
            )

            server = result.scalar_one_or_none()

            if server is None:
                return None

            return self._to_server_info(server)

    async def get_server_by_name(
        self,
        name: str,
    ) -> ToolServerInfo | None:
        """根据 Server 名称查询配置。"""

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ToolServer).where(
                    ToolServer.name == name
                )
            )

            server = result.scalar_one_or_none()

            if server is None:
                return None

            return self._to_server_info(server)

    async def list_servers(
        self,
    ) -> list[ToolServerInfo]:
        """查询所有 Server 配置。"""

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ToolServer).order_by(
                    ToolServer.created_at.desc()
                )
            )

            servers = result.scalars().all()

            return [
                self._to_server_info(server)
                for server in servers
            ]

    async def update_server_status(
        self,
        server_id: str,
        status: str,
    ) -> bool:
        """
        更新 Server 连接状态。

        例如：
        connected / disconnected / error / disabled
        """

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                update(ToolServer)
                .where(ToolServer.server_id == server_id)
                .values(status=status)
            )

            await session.commit()

            return result.rowcount > 0

    # =========================================================
    # 二、工具目录查询
    # =========================================================

    async def list_tool_summaries(
        self,
        query: ToolDiscoveryQuery,
    ) -> list[CatalogToolSummary]:
        """
        查询工具轻量信息。

        重要：
        这里不加载 input_schema 和 metadata。

        后续 Tool Discovery Engine 会优先调用此方法，
        避免每次搜索都读取所有工具的完整参数 Schema。
        """

        async with AsyncSessionLocal() as session:
            statement = select(
                CatalogTool.tool_id,
                CatalogTool.server_id,
                CatalogTool.provider_type,
                CatalogTool.name,
                CatalogTool.description,
                CatalogTool.capabilities,
                CatalogTool.keywords,
                CatalogTool.status,
            ).where(
                CatalogTool.status == "active"
            )

            # 按 Provider 类型过滤
            if query.provider_types:
                provider_values = [
                    item.value if hasattr(item, "value") else item
                    for item in query.provider_types
                ]

                statement = statement.where(
                    CatalogTool.provider_type.in_(provider_values)
                )

            # 按能力标签过滤
            # JSONB contains：工具必须包含指定的全部能力标签
            if query.capabilities:
                statement = statement.where(
                    CatalogTool.capabilities.contains(
                        query.capabilities
                    )
                )

            # 关键词过滤
            # 当前采用名称和描述的模糊匹配。
            # 后续可以扩展 PostgreSQL 全文检索或向量检索。
            if query.query:
                keyword = f"%{query.query}%"

                statement = statement.where(
                    or_(
                        CatalogTool.name.ilike(keyword),
                        CatalogTool.description.ilike(keyword),
                    )
                )

            if query.keywords:
                keyword_conditions = []

                for keyword in query.keywords:
                    pattern = f"%{keyword}%"

                    keyword_conditions.extend([
                        CatalogTool.name.ilike(pattern),
                        CatalogTool.description.ilike(pattern),
                    ])

                statement = statement.where(
                    or_(*keyword_conditions)
                )

            statement = statement.order_by(
                CatalogTool.updated_at.desc()
            ).limit(query.top_k)

            result = await session.execute(statement)

            rows = result.mappings().all()

            return [
                CatalogToolSummary.model_validate(dict(row))
                for row in rows
            ]

    async def get_tool_detail(
        self,
        tool_id: str,
    ) -> CatalogToolDetail | None:
        """
        根据 tool_id 加载工具完整定义。

        只有工具被 Discovery 选中后，
        才调用此方法加载 input_schema。
        """

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(CatalogTool).where(
                    CatalogTool.tool_id == tool_id
                )
            )

            tool = result.scalar_one_or_none()

            if tool is None:
                return None

            return CatalogToolDetail(
                tool_id=tool.tool_id,
                server_id=tool.server_id,
                provider_type=tool.provider_type,
                name=tool.name,
                description=tool.description,
                capabilities=tool.capabilities,
                keywords=tool.keywords,
                status=tool.status,
                input_schema=tool.input_schema,
                metadata=tool.tool_metadata,
            )

    # =========================================================
    # 三、工具目录同步
    # =========================================================

    async def sync_server_tools(
        self,
        server_id: str,
        tools: list[CatalogToolCreate],
    ) -> dict[str, int]:
        """
        同步某个 Server 当前提供的工具列表。

        同步规则：

        1. 新工具：插入数据库。
        2. 已存在工具：更新描述、能力、Schema 等信息。
        3. 已经不存在的旧工具：标记为 unavailable。
        4. 不物理删除旧工具。

        注意：
        该方法只同步指定 server_id 的工具，
        不会影响其他 MCP Server。
        """

        async with AsyncSessionLocal() as session:
            # 确认 Server 存在
            server_result = await session.execute(
                select(ToolServer).where(
                    ToolServer.server_id == server_id
                )
            )

            server = server_result.scalar_one_or_none()

            if server is None:
                raise ValueError(
                    f"Tool Server 不存在: {server_id}"
                )

            provider_type = server.provider_type

            # 本次同步中出现的 source_key
            current_source_keys: set[str] = set()

            created_count = 0
            updated_count = 0

            for item in tools:
                # 不允许将工具同步到其他 Server
                if item.server_id != server_id:
                    raise ValueError(
                        f"工具 {item.name} 的 server_id "
                        f"与当前同步目标不一致"
                    )

                item_provider = (
                    item.provider_type.value
                    if hasattr(item.provider_type, "value")
                    else item.provider_type
                )

                if item_provider != provider_type:
                    raise ValueError(
                        f"工具 {item.name} 的 provider_type "
                        f"与 Server 类型不一致"
                    )

                # source_key 区分不同来源的同名工具
                source_key = (
                    f"{provider_type}:{server_id}:{item.name}"
                )

                current_source_keys.add(source_key)

                result = await session.execute(
                    select(CatalogTool).where(
                        CatalogTool.source_key == source_key
                    )
                )

                existing = result.scalar_one_or_none()

                if existing is None:
                    # 新工具：创建目录记录
                    catalog_tool = CatalogTool(
                        server_id=server_id,
                        provider_type=provider_type,
                        name=item.name,
                        description=item.description,
                        capabilities=item.capabilities or [],
                        keywords=item.keywords or [],
                        input_schema=item.input_schema or {},
                        tool_metadata=getattr(
                            item,
                            "metadata",
                            {},
                        ) or {},
                        status=(
                            item.status.value
                            if hasattr(item.status, "value")
                            else item.status
                        ),
                        source_key=source_key,
                    )

                    session.add(catalog_tool)
                    created_count += 1

                else:
                    # 已有工具：更新目录信息
                    existing.description = item.description
                    existing.capabilities = item.capabilities or []
                    existing.keywords = item.keywords or []
                    existing.input_schema = item.input_schema or {}
                    existing.tool_metadata = getattr(
                        item,
                        "metadata",
                        {},
                    ) or {}

                    # 管理员手动禁用的工具，不允许被自动同步重新启用。
                    # 其他状态按照 Server 当前返回的工具定义更新。
                    if existing.status != "disabled":
                        existing.status = (
                            item.status.value
                            if hasattr(item.status, "value")
                            else item.status
                        )

                    updated_count += 1

            # 将本次同步中没有出现的旧工具标记为 unavailable
            stale_statement = update(CatalogTool).where(
                CatalogTool.server_id == server_id,
                CatalogTool.status != "deleted",
            )

            if current_source_keys:
                stale_statement = stale_statement.where(
                    CatalogTool.source_key.notin_(
                        current_source_keys
                    )
                )

            stale_statement = stale_statement.values(
                status="unavailable"
            )

            stale_result = await session.execute(stale_statement)

            unavailable_count = stale_result.rowcount or 0

            # 同步完成后更新 Server 状态
            server.status = "connected"

            # 整个同步过程统一提交
            await session.commit()

            return {
                "created": created_count,
                "updated": updated_count,
                "unavailable": unavailable_count,
                "total": len(tools),
            }

    # =========================================================
    # 四、工具状态管理
    # =========================================================

    async def update_tool_status(
        self,
        tool_id: str,
        status: str,
    ) -> bool:
        """启用、禁用或调整单个工具状态。"""

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                update(CatalogTool)
                .where(CatalogTool.tool_id == tool_id)
                .values(status=status)
            )

            await session.commit()

            return result.rowcount > 0

    # =========================================================
    # 五、内部转换方法
    # =========================================================

    @staticmethod
    def _to_server_info(
        server: ToolServer,
    ) -> ToolServerInfo:
        """将 ORM 对象转换为 Pydantic DTO。"""

        return ToolServerInfo(
            server_id=server.server_id,
            name=server.name,
            provider_type=server.provider_type,
            server_url=server.server_url,
            description=server.description,
            status=server.status,
            authentication_ref=server.authentication_ref,
            connection_config=server.connection_config or {},
        )


# 全局 Repository 实例
tool_catalog_repository = ToolCatalogRepository()
