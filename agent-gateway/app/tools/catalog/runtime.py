from app.tools.catalog.adapter import MCPToolAdapter
from app.tools.catalog.service import tool_catalog_service
from app.tools.registry import tool_registry
from app.tools.providers.mcp import mcp_provider


# MCP 动态工具适配器
mcp_tool_adapter = MCPToolAdapter(
    catalog_service=tool_catalog_service,
    tool_registry=tool_registry,
    mcp_provider=mcp_provider,
)