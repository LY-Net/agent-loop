
import asyncio
import uuid

from sqlalchemy import delete, select

from app.main import app
from app.db.session import AsyncSessionLocal
from app.models.tool_catalog import CatalogTool, ToolServer
from app.models.tool_embedding import ToolEmbedding
from app.tools.catalog.repository import ToolCatalogRepository
from app.tools.catalog.service import ToolCatalogService
from app.tools.catalog.schemas import (
    ToolProviderType,
    ToolServerCreate,
)


def log_pass(message: str) -> None:
    print(f"[PASS] {message}")


class FakeMCPProvider:
    """模拟 MCP Server，只模拟工具发现，不模拟 Embedding。"""

    def __init__(self):
        self.tools = []

    async def list_tools(
        self,
        server_url: str,
        transport: str = "streamable_http",
    ):
        return self.tools


def make_tool(
    name: str,
    description: str,
) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
        },
    }


async def get_tool(
    tool_id: str,
) -> CatalogTool | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(CatalogTool).where(
                CatalogTool.tool_id == tool_id
            )
        )
        return result.scalar_one_or_none()


async def get_embedding(
    tool_id: str,
) -> ToolEmbedding | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ToolEmbedding).where(
                ToolEmbedding.tool_id == tool_id,
                ToolEmbedding.space_id == "tool_discovery",
            )
        )
        return result.scalar_one_or_none()


async def main() -> None:

    suffix = uuid.uuid4().hex[:8]
    server_name = f"test-tool-embedding-{suffix}"
    server_id = None

    print("=" * 68)
    print("TOOL CATALOG + EMBEDDING INTEGRATION TEST")
    print("=" * 68)

    async with app.router.lifespan_context(app):

        provider = FakeMCPProvider()

        async with AsyncSessionLocal() as session:

            repository = ToolCatalogRepository(
                session=session
            )

            service = ToolCatalogService(
                repository=repository,
                embedding_space_service=(
                    app.state.embedding_space_service
                ),
                provider=provider,
            )

            try:
                # -------------------------------------------------
                # 1. 创建测试 Server
                # -------------------------------------------------

                server = await service.register_server(
                    ToolServerCreate(
                        name=server_name,
                        provider_type=ToolProviderType.MCP,
                        server_url="http://127.0.0.1:9001/mcp",
                        description="Tool Embedding Integration Test",
                    )
                )

                server_id = server.server_id

                log_pass("测试 MCP Server 创建成功")

                # -------------------------------------------------
                # 2. 首次同步：两个工具
                # -------------------------------------------------

                provider.tools = [
                    make_tool(
                        "lookup_customer",
                        "根据客户名称或客户编号查询客户信息",
                    ),
                    make_tool(
                        "query_orders",
                        "根据订单编号查询订单详情和订单状态",
                    ),
                ]

                result = await service.sync_server_tools(
                    server_id
                )

                assert result["created"] == 2
                assert result["embedding_count"] == 2
                assert result["deleted_embedding_count"] == 0

                alpha = await session.execute(
                    select(CatalogTool).where(
                        CatalogTool.server_id == server_id,
                        CatalogTool.name == "lookup_customer",
                    )
                )
                alpha = alpha.scalar_one()

                beta = await session.execute(
                    select(CatalogTool).where(
                        CatalogTool.server_id == server_id,
                        CatalogTool.name == "query_orders",
                    )
                )
                beta = beta.scalar_one()

                alpha_id = alpha.tool_id
                beta_id = beta.tool_id

                alpha_embedding = await get_embedding(alpha_id)
                beta_embedding = await get_embedding(beta_id)

                assert alpha_embedding is not None
                assert beta_embedding is not None
                assert len(alpha_embedding.embedding) == 1024
                assert len(beta_embedding.embedding) == 1024

                log_pass("首次同步：Catalog 和 1024 维向量均已持久化")

                # -------------------------------------------------
                # 3. 重复同步：不重复创建
                # -------------------------------------------------

                result = await service.sync_server_tools(
                    server_id
                )

                assert result["created"] == 0
                assert result["updated"] == 2
                assert result["embedding_count"] == 2

                alpha_after = await get_tool(alpha_id)

                assert alpha_after is not None
                assert alpha_after.tool_id == alpha_id

                log_pass("重复同步：工具 ID 保持不变，向量执行更新")

                # -------------------------------------------------
                # 4. 修改描述：验证向量文本更新
                # -------------------------------------------------

                old_embedding = await get_embedding(alpha_id)
                old_text = old_embedding.embedding_text

                provider.tools = [
                    make_tool(
                        "lookup_customer",
                        "根据手机号、客户名称查询客户档案和联系方式",
                    ),
                ]

                result = await service.sync_server_tools(
                    server_id
                )

                assert result["unavailable"] == 1
                assert result["embedding_count"] == 1
                assert result["deleted_embedding_count"] == 1

                new_embedding = await get_embedding(alpha_id)

                assert new_embedding is not None
                assert new_embedding.embedding_text != old_text
                assert await get_embedding(beta_id) is None

                beta_after = await get_tool(beta_id)

                assert beta_after is not None
                assert beta_after.status == "unavailable"

                log_pass("工具描述更新成功，失效工具向量已清理")

                # -------------------------------------------------
                # 5. 管理员禁用状态保护
                # -------------------------------------------------

                async with session.begin():
                    await repository.update_tool_status(
                        alpha_id,
                        "disabled",
                    )

                provider.tools = [
                    make_tool(
                        "lookup_customer",
                        "查询客户资料",
                    ),
                ]

                result = await service.sync_server_tools(
                    server_id
                )

                alpha_after = await get_tool(alpha_id)

                assert alpha_after is not None
                assert alpha_after.status == "disabled"
                assert await get_embedding(alpha_id) is None

                log_pass("管理员禁用状态得到保留，禁用工具向量已清理")

                # -------------------------------------------------
                # 6. Embedding 失败：Catalog 不应发生变化
                # -------------------------------------------------

                async with session.begin():
                    await repository.update_tool_status(
                        alpha_id,
                        "active",
                    )

                provider.tools = [
                    make_tool(
                        "lookup_customer",
                        "原有客户查询描述",
                    ),
                ]

                await service.sync_server_tools(server_id)

                old_tool = await get_tool(alpha_id)
                old_description = old_tool.description

                original_embed = (
                    service.embedding_space_service.embed
                )

                async def failed_embed(*args, **kwargs):
                    raise RuntimeError(
                        "模拟 Embedding 服务故障"
                    )

                service.embedding_space_service.embed = failed_embed

                embedding_failed = False

                try:
                    provider.tools = [
                        make_tool(
                            "lookup_customer",
                            "这次不应该写入数据库的新描述",
                        ),
                    ]

                    await service.sync_server_tools(server_id)

                except RuntimeError:
                    embedding_failed = True

                finally:
                    service.embedding_space_service.embed = (
                        original_embed
                    )

                assert embedding_failed

                tool_after_failure = await get_tool(alpha_id)

                assert tool_after_failure.description == old_description

                log_pass("Embedding 生成失败：Catalog 未发生修改")

                # -------------------------------------------------
                # 7. 数据库写入失败：事务整体回滚
                # -------------------------------------------------

                original_upsert = (
                    service.embedding_repository.upsert_many
                )

                async def failed_upsert(items):
                    raise RuntimeError(
                        "模拟 Tool Embedding 数据库写入失败"
                    )

                service.embedding_repository.upsert_many = failed_upsert

                database_failed = False

                try:
                    provider.tools = [
                        make_tool(
                            "lookup_customer",
                            "模拟数据库失败的新工具描述",
                        ),
                    ]

                    await service.sync_server_tools(server_id)

                except RuntimeError:
                    database_failed = True

                finally:
                    service.embedding_repository.upsert_many = (
                        original_upsert
                    )

                assert database_failed

                tool_after_rollback = await get_tool(alpha_id)

                assert tool_after_rollback.description == old_description

                embedding_after_rollback = await get_embedding(
                    alpha_id
                )

                assert embedding_after_rollback is not None

                log_pass("Embedding 写入失败：Catalog 和向量事务回滚")

                print()
                print("=" * 68)
                print("RESULT: ALL PASSED")
                print("=" * 68)

            finally:
                # -------------------------------------------------
                # 8. 清理测试数据
                # -------------------------------------------------

                if server_id:
                    if session.in_transaction():
                        await session.rollback()

                    async with session.begin():
                        await session.execute(
                            delete(ToolServer).where(
                                ToolServer.server_id == server_id
                            )
                        )

                    print(
                        "[CLEANUP] 测试 Server、Catalog 和 Embedding 已清理"
                    )


if __name__ == "__main__":
    asyncio.run(main())
