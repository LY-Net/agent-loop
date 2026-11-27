
import asyncio
import uuid

from sqlalchemy import delete, select

from app.db.session import AsyncSessionLocal
from app.models.tool_catalog import CatalogTool, ToolServer
from app.tools.catalog.repository import ToolCatalogRepository
from app.tools.catalog.schemas import (
    CatalogToolCreate,
    ToolDiscoveryQuery,
    ToolProviderType,
    ToolServerCreate,
)


def log_pass(message: str) -> None:
    print(f"[PASS] {message}")


def create_tool(
    server_id: str,
    name: str,
    description: str = "",
) -> CatalogToolCreate:
    return CatalogToolCreate(
        server_id=server_id,
        provider_type=ToolProviderType.MCP,
        name=name,
        description=description or f"{name} tool",
        capabilities=["test", "database_read"],
        keywords=[name, "test"],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
        },
        metadata={"test": True},
    )


async def get_catalog_tool(
    source_key: str,
) -> CatalogTool | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(CatalogTool).where(
                CatalogTool.source_key == source_key
            )
        )
        return result.scalar_one_or_none()


async def main() -> None:
    repository = ToolCatalogRepository()

    suffix = uuid.uuid4().hex[:8]

    server_ids: list[str] = []

    print("=" * 65)
    print("TOOL CATALOG REPOSITORY INTEGRATION TEST")
    print("=" * 65)

    try:
        # -------------------------------------------------
        # 1. 创建 MCP Server
        # -------------------------------------------------
        server_a = await repository.create_server(
            ToolServerCreate(
                name=f"test-mcp-a-{suffix}",
                provider_type=ToolProviderType.MCP,
                server_url="http://127.0.0.1:9001/mcp",
                description="Tool Catalog Integration Test A",
            )
        )
        server_ids.append(server_a.server_id)

        assert server_a.status.value == "pending" or (
            server_a.status == "pending"
        )

        log_pass("MCP Server 创建成功")

        # -------------------------------------------------
        # 2. 首次同步三个工具
        # -------------------------------------------------
        initial_tools = [
            create_tool(
                server_a.server_id,
                "lookup_customer",
            ),
            create_tool(
                server_a.server_id,
                "query_orders",
            ),
            create_tool(
                server_a.server_id,
                "create_ticket",
            ),
        ]

        result = await repository.sync_server_tools(
            server_a.server_id,
            initial_tools,
        )

        assert result["created"] == 3
        assert result["updated"] == 0
        assert result["total"] == 3

        log_pass("首次同步：新增 3 个工具")

        # -------------------------------------------------
        # 3. 重复同步
        # -------------------------------------------------
        result = await repository.sync_server_tools(
            server_a.server_id,
            initial_tools,
        )

        assert result["created"] == 0
        assert result["updated"] == 3

        log_pass("重复同步：更新 3 个工具，没有重复新增")

        # -------------------------------------------------
        # 4. 模拟工具列表变化
        #    query_orders 被移除
        #    export_report 被新增
        # -------------------------------------------------
        changed_tools = [
            create_tool(
                server_a.server_id,
                "lookup_customer",
            ),
            create_tool(
                server_a.server_id,
                "create_ticket",
            ),
            create_tool(
                server_a.server_id,
                "export_report",
            ),
        ]

        result = await repository.sync_server_tools(
            server_a.server_id,
            changed_tools,
        )

        assert result["created"] == 1
        assert result["updated"] == 2
        assert result["unavailable"] == 1

        log_pass("工具列表变化：新增 1 个，失效 1 个")

        # -------------------------------------------------
        # 5. 管理员禁用状态保护
        # -------------------------------------------------
        lookup_source_key = (
            f"mcp:{server_a.server_id}:lookup_customer"
        )

        lookup_tool = await get_catalog_tool(
            lookup_source_key
        )

        assert lookup_tool is not None

        updated = await repository.update_tool_status(
            lookup_tool.tool_id,
            "disabled",
        )

        assert updated is True

        # 再次同步，Server 仍然返回 lookup_customer
        result = await repository.sync_server_tools(
            server_a.server_id,
            changed_tools,
        )

        lookup_detail = await repository.get_tool_detail(
            lookup_tool.tool_id
        )

        assert lookup_detail is not None
        assert lookup_detail.status.value == "disabled" or (
            lookup_detail.status == "disabled"
        )

        log_pass("管理员禁用状态不会被自动同步覆盖")

        # -------------------------------------------------
        # 6. Summary / Detail 按需加载
        # -------------------------------------------------
        summaries = await repository.list_tool_summaries(
            ToolDiscoveryQuery(top_k=100)
        )

        assert summaries

        # Summary 不应携带完整 Schema
        assert all(
            not hasattr(item, "input_schema")
            for item in summaries
        )

        # Detail 应包含完整 Schema
        detail = await repository.get_tool_detail(
            lookup_tool.tool_id
        )

        assert detail is not None
        assert detail.input_schema
        assert "properties" in detail.input_schema

        log_pass("Summary 轻量查询与 Detail 完整 Schema 加载正常")

        # -------------------------------------------------
        # 7. 不同 Server 允许存在同名工具
        # -------------------------------------------------
        server_b = await repository.create_server(
            ToolServerCreate(
                name=f"test-mcp-b-{suffix}",
                provider_type=ToolProviderType.MCP,
                server_url="http://127.0.0.1:9002/mcp",
                description="Tool Catalog Integration Test B",
            )
        )
        server_ids.append(server_b.server_id)

        await repository.sync_server_tools(
            server_b.server_id,
            [
                create_tool(
                    server_b.server_id,
                    "create_ticket",
                )
            ],
        )

        source_key_b = (
            f"mcp:{server_b.server_id}:create_ticket"
        )

        tool_b = await get_catalog_tool(source_key_b)

        source_key_a = (
            f"mcp:{server_a.server_id}:create_ticket"
        )

        tool_a = await get_catalog_tool(source_key_a)

        assert tool_a is not None
        assert tool_b is not None
        assert tool_a.tool_id != tool_b.tool_id

        log_pass("不同 Server 的同名工具可以独立存在")

        # -------------------------------------------------
        # 8. 异常回滚
        # -------------------------------------------------
        server_c = await repository.create_server(
            ToolServerCreate(
                name=f"test-mcp-c-{suffix}",
                provider_type=ToolProviderType.MCP,
                server_url="http://127.0.0.1:9003/mcp",
                description="Tool Catalog Rollback Test",
            )
        )
        server_ids.append(server_c.server_id)

        rollback_source_key = (
            f"mcp:{server_c.server_id}:rollback_probe"
        )

        rollback_failed = False

        try:
            await repository.sync_server_tools(
                server_c.server_id,
                [
                    create_tool(
                        server_c.server_id,
                        "rollback_probe",
                    ),
                    create_tool(
                        "wrong-server-id",
                        "invalid_tool",
                    ),
                ],
            )
        except ValueError:
            rollback_failed = True

        assert rollback_failed

        rollback_tool = await get_catalog_tool(
            rollback_source_key
        )

        assert rollback_tool is None

        server_c_after = await repository.get_server(
            server_c.server_id
        )

        assert server_c_after is not None
        assert (
            server_c_after.status.value == "pending"
            or server_c_after.status == "pending"
        )

        log_pass("同步异常时事务回滚，没有留下部分数据")

        print()
        print("=" * 65)
        print("RESULT: ALL PASSED")
        print("=" * 65)

    finally:
        # -------------------------------------------------
        # 清理本次测试数据
        # -------------------------------------------------
        if server_ids:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    delete(ToolServer).where(
                        ToolServer.server_id.in_(server_ids)
                    )
                )
                await session.commit()

            print("[CLEANUP] 测试 Server 及关联工具已清理")


if __name__ == "__main__":
    asyncio.run(main())
