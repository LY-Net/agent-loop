
import asyncio

from app.tools.catalog.service import tool_catalog_service
from app.tools.catalog.schemas import ToolDiscoveryQuery


async def main():
    print("\n========== CATALOG DISCOVERY TEST ==========")

    # 模拟用户提出的自然语言问题
    user_query = "帮我检索人工智能 Agent 相关的学术文献"

    # 构造轻量检索请求
    discovery_query = ToolDiscoveryQuery(
        query=user_query,
        provider_types=["mcp"],
        top_k=5,
    )

    # 只检索 Catalog 轻量元数据
    results = await tool_catalog_service.discover_tools(
        discovery_query
    )

    print("\n[1] Catalog 检索完成")
    print(f"候选工具数量: {len(results)}")

    # 输出候选工具的轻量信息
    for index, tool in enumerate(results, start=1):
        print(f"\n候选工具 {index}")
        print(f"tool_id: {tool.tool_id}")
        print(f"name: {tool.name}")
        print(f"description: {tool.description}")
        print(f"provider_type: {tool.provider_type}")
        print(f"capabilities: {tool.capabilities}")
        print(f"keywords: {tool.keywords}")

    print("\n========== DISCOVERY TEST FINISHED ==========")


if __name__ == "__main__":
    asyncio.run(main())
