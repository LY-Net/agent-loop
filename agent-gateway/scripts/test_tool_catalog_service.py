
import asyncio
import uuid
from types import SimpleNamespace
from typing import Any

from app.tools.catalog.schemas import (
    ToolProviderType,
    ToolServerStatus,
    ToolServerCreate,
)
from app.tools.catalog.service import ToolCatalogService


# ============================================================
# Mock Repository
# 模拟数据库行为，验证 Service 是否通过 server_id 获取配置
# ============================================================

class MockToolCatalogRepository:

    def __init__(self):
        self.servers = {}
        self.synced_tools = {}
        self.server_status = {}

    async def get_server_by_name(self, name: str):
        for server in self.servers.values():
            if server.name == name:
                return server
        return None

    async def create_server(self, data):
        server_id = str(uuid.uuid4())

        server = SimpleNamespace(
            server_id=server_id,
            name=data.name,
            provider_type=data.provider_type,
            server_url=str(data.server_url),
            description=data.description,
            authentication_ref=data.authentication_ref,
            connection_config=data.connection_config,
            status=ToolServerStatus.PENDING,
        )

        self.servers[server_id] = server
        self.server_status[server_id] = ToolServerStatus.PENDING

        return server

    async def get_server(self, server_id: str):
        return self.servers.get(server_id)

    async def update_server_status(self, server_id: str, status):
        if server_id not in self.servers:
            raise ValueError("Server 不存在")

        self.server_status[server_id] = status
        self.servers[server_id].status = status
        return True

    async def sync_server_tools(self, server_id: str, tools):
        self.synced_tools[server_id] = tools

        return {
            "created": len(tools),
            "updated": 0,
            "unavailable": 0,
        }


# ============================================================
# Mock MCP Provider
# 模拟 MCP Server 返回的 tools/list 结果
# ============================================================

class MockMCPProvider:

    def __init__(self):
        self.called_urls = []

    async def list_tools(self, server_url: str) -> list[dict[str, Any]]:
        self.called_urls.append(server_url)

        return [
            {
                "name": "query_sales",
                "description": "查询销售数据",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "region": {
                            "type": "string",
                            "description": "地区",
                        }
                    },
                    "required": ["region"],
                },
            },
            {
                "name": "query_customers",
                "description": "查询客户数据",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                        }
                    },
                },
            },
        ]


# ============================================================
# Test
# ============================================================

async def test_config_driven_mcp_server():

    repository = MockToolCatalogRepository()
    provider = MockMCPProvider()

    service = ToolCatalogService(
        repository=repository,
        provider=provider,
    )

    # 测试配置：URL 只存在于注册配置中，不写入 Service
    config = ToolServerCreate(
        name="test-mcp-server",
        provider_type=ToolProviderType.MCP,
        server_url="http://127.0.0.1:9000/mcp",
        description="配置驱动测试 MCP Server",
        authentication_ref=None,
        connection_config={},
    )

    # --------------------------------------------------------
    # 1. 注册 Server
    # --------------------------------------------------------

    server = await service.register_server(config)

    server_id = server.server_id

    assert server_id in repository.servers

    # 注册时不应该调用 MCP Provider
    assert len(provider.called_urls) == 0

    print("[PASS] Server 注册成功，注册过程没有建立连接")

    # --------------------------------------------------------
    # 2. 根据 server_id 连接
    # --------------------------------------------------------

    result = await service.connect_server(server_id)

    assert result["success"] is True

    # Provider 使用的是数据库中保存的 URL
    assert provider.called_urls == [
        repository.servers[server_id].server_url
    ]

    print("[PASS] connect_server 根据 server_id 自动读取配置")

    # --------------------------------------------------------
    # 3. 检查工具同步结果
    # --------------------------------------------------------

    tools = repository.synced_tools[server_id]

    assert len(tools) == 2

    tool_names = {tool.name for tool in tools}

    assert tool_names == {
        "query_sales",
        "query_customers",
    }

    print("[PASS] MCP 工具自动发现并同步到 Catalog")

    # --------------------------------------------------------
    # 4. 检查连接状态
    # --------------------------------------------------------

    assert (
        repository.server_status[server_id]
        == ToolServerStatus.CONNECTED
    )

    print("[PASS] Server 状态更新为 CONNECTED")

    # --------------------------------------------------------
    # 5. 断开 Server
    # --------------------------------------------------------

    disconnected = await service.disconnect_server(server_id)

    assert disconnected is True

    assert (
        repository.server_status[server_id]
        == ToolServerStatus.DISCONNECTED
    )

    print("[PASS] Server 状态更新为 DISCONNECTED")

    print("\n========== ALL TESTS PASSED ==========")


if __name__ == "__main__":
    asyncio.run(test_config_driven_mcp_server())
