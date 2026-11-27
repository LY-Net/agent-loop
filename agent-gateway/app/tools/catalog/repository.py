
"""
Tool Catalog Repository

职责：
1. 管理 Tool Server 配置。
2. 管理工具目录元数据。
3. 提供轻量工具列表查询。
4. 按需加载完整工具 Schema。
5. 支持 MCP Server 工具列表的批量同步。

设计原则：
- Repository 只负责数据库操作。
- AsyncSession 由外部注入。
- Repository 不创建 Session。
- Repository 不主动 commit。
- 事务统一由 Service 控制。
- 同步失效的工具标记为 unavailable，不物理删除。
"""

from sqlalchemy import select, update, or_
from sqlalchemy.ext.asyncio import AsyncSession

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

    使用外部传入的 AsyncSession。
    所有写入操作只 flush，不 commit。
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # =========================================================
    # 一、Tool Server 管理
    # =========================================================

    async def create_server(
        self,
        data: ToolServerCreate,
    ) -> ToolServerInfo:
        server = ToolServer(
            name=data.name,
            provider_type=(
                data.provider_type.value
                if hasattr(data.provider_type, "value")
                else data.provider_type
            ),
            server_url=(
                str(data.server_url)
                if data.server_url
                else None
            ),
            description=data.description,
            authentication_ref=data.authentication_ref,
            connection_config=data.connection_config or {},
        )

        self.session.add(server)
        await self.session.flush()
        await self.session.refresh(server)

        return self._to_server_info(server)

    async def get_server(
        self,
        server_id: str,
    ) -> ToolServerInfo | None:
        result = await self.session.execute(
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
        result = await self.session.execute(
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
        result = await self.session.execute(
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
        result = await self.session.execute(
            update(ToolServer)
            .where(ToolServer.server_id == server_id)
            .values(status=status)
        )

        return result.rowcount > 0

    # =========================================================
    # 二、工具目录查询
    # =========================================================

    async def list_tool_summaries(
        self,
        query: ToolDiscoveryQuery,
    ) -> list[CatalogToolSummary]:

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

        # Provider 类型过滤
        if query.provider_types:
            provider_values = [
                item.value if hasattr(item, "value") else item
                for item in query.provider_types
            ]

            statement = statement.where(
                CatalogTool.provider_type.in_(provider_values)
            )

        # 能力标签过滤
        if query.capabilities:
            statement = statement.where(
                CatalogTool.capabilities.contains(
                    query.capabilities
                )
            )

        # 自然语言/关键词模糊匹配
        if query.query:
            keyword = f"%{query.query}%"

            statement = statement.where(
                or_(
                    CatalogTool.name.ilike(keyword),
                    CatalogTool.description.ilike(keyword),
                )
            )

        # 多关键词匹配
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

        result = await self.session.execute(statement)

        rows = result.mappings().all()

        return [
            CatalogToolSummary.model_validate(dict(row))
            for row in rows
        ]

    async def get_tool_detail(
        self,
        tool_id: str,
    ) -> CatalogToolDetail | None:

        result = await self.session.execute(
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
    ) -> dict:
        """
        同步某个 Server 当前提供的工具列表。

        返回：
        {
            "created": int,
            "updated": int,
            "unavailable": int,
            "total": int,
            "tools": list[CatalogToolSummary]
        }

        注意：
        1. 不 commit，由外部 Service 控制事务。
        2. flush 后返回工具 ID，供后续 Embedding 写入使用。
        3. 管理员禁用的工具不被自动重新启用。
        """

        # 1. 确认 Server 存在
        server_result = await self.session.execute(
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

        current_source_keys: set[str] = set()

        created_count = 0
        updated_count = 0

        # 2. 逐个新增或更新
        for item in tools:

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

            source_key = (
                f"{provider_type}:{server_id}:{item.name}"
            )

            current_source_keys.add(source_key)

            result = await self.session.execute(
                select(CatalogTool).where(
                    CatalogTool.source_key == source_key
                )
            )

            existing = result.scalar_one_or_none()

            item_status = (
                item.status.value
                if hasattr(item.status, "value")
                else item.status
            )

            if existing is None:
                catalog_tool = CatalogTool(
                    server_id=server_id,
                    provider_type=provider_type,
                    name=item.name,
                    description=item.description,
                    capabilities=item.capabilities or [],
                    keywords=item.keywords or [],
                    input_schema=item.input_schema or {},
                    tool_metadata=item.metadata or {},
                    status=item_status,
                    source_key=source_key,
                )

                self.session.add(catalog_tool)
                created_count += 1

            else:
                existing.description = item.description
                existing.capabilities = item.capabilities or []
                existing.keywords = item.keywords or []
                existing.input_schema = item.input_schema or {}
                existing.tool_metadata = item.metadata or {}

                # 管理员禁用状态优先级最高
                if existing.status != "disabled":
                    existing.status = item_status

                updated_count += 1

        # 3. 先 flush，确保新工具获得 tool_id
        await self.session.flush()

        # 4. 找出本次同步已经不存在的工具
        stale_statement = update(CatalogTool).where(
            CatalogTool.server_id == server_id,
            CatalogTool.status.notin_([
                "deleted",
                "disabled",
            ]),
        )

        if current_source_keys:
            stale_statement = stale_statement.where(
                CatalogTool.source_key.notin_(
                    current_source_keys
                )
            )

        stale_result = await self.session.execute(
            stale_statement.values(
                status="unavailable"
            )
        )

        unavailable_count = stale_result.rowcount or 0

        # 5. 更新 Server 状态
        server.status = "connected"

        await self.session.flush()

        # 6. 重新读取本 Server 的工具，返回 tool_id
        tools_result = await self.session.execute(
            select(
                CatalogTool.tool_id,
                CatalogTool.server_id,
                CatalogTool.provider_type,
                CatalogTool.name,
                CatalogTool.description,
                CatalogTool.capabilities,
                CatalogTool.keywords,
                CatalogTool.status,
            ).where(
                CatalogTool.server_id == server_id,
                CatalogTool.status != "deleted",
            ).order_by(
                CatalogTool.name
            )
        )

        rows = tools_result.mappings().all()

        tool_summaries = [
            CatalogToolSummary.model_validate(dict(row))
            for row in rows
        ]

        return {
            "created": created_count,
            "updated": updated_count,
            "unavailable": unavailable_count,
            "total": len(tools),
            "tools": tool_summaries,
        }

    # =========================================================
    # 四、工具状态管理
    # =========================================================

    async def update_tool_status(
        self,
        tool_id: str,
        status: str,
    ) -> bool:

        result = await self.session.execute(
            update(CatalogTool)
            .where(CatalogTool.tool_id == tool_id)
            .values(status=status)
        )

        return result.rowcount > 0

    # =========================================================
    # 五、内部转换方法
    # =========================================================

    @staticmethod
    def _to_server_info(
        server: ToolServer,
    ) -> ToolServerInfo:

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
