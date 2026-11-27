
"""
MCP Provider

基于 MCP Python SDK 实现远程 MCP Server 接入。

当前使用：
Streamable HTTP Transport

实现能力：
1. 通过 Server URL 连接 MCP Server。
2. 自动初始化 MCP 会话。
3. 获取 MCP Server 提供的工具列表。
4. 调用远程 MCP 工具。
5. 将 MCP 原生响应转换为统一 Python 数据结构。

当前版本暂时不处理认证凭证。
后续通过 CredentialResolver 注入认证信息，
避免在代码或数据库中保存明文 Token。
"""

from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.tools.providers.base import BaseToolProvider
from mcp.client.sse import sse_client

class MCPProvider(BaseToolProvider):
    """远程 MCP Server Provider。"""

    @staticmethod
    def _validate_url(server_url: str) -> None:
        """
        校验 MCP Server URL。

        目前只接受 HTTP 和 HTTPS。
        生产环境建议使用 HTTPS。
        """

        if not server_url:
            raise ValueError("MCP Server URL 不能为空")

        if not server_url.startswith(("http://", "https://")):
            raise ValueError(
                f"不支持的 MCP Server URL: {server_url}"
            )

    @staticmethod
    def _normalize_tool(tool: Any) -> dict[str, Any]:
        """
        将 MCP SDK 返回的 Tool 对象转换为 Gateway 统一格式。

        兼容 SDK 不同版本的字段命名：
        - input_schema
        - inputSchema
        """

        # 兼容 Pydantic 模型和普通字典
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description")
            input_schema = (
                tool.get("input_schema")
                or tool.get("inputSchema")
                or {}
            )
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)

            input_schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {}
            )

        if not name:
            raise ValueError("MCP Tool 缺少 name 字段")

        return {
            "name": name,
            "description": description or "",
            "input_schema": input_schema,
        }

    @staticmethod
    def _normalize_result(result: Any) -> dict[str, Any]:
        """
        将 MCP CallToolResult 转换为统一结果格式。
        """

        if isinstance(result, dict):
            structured_content = (
                    result.get("structured_content")
                    or result.get("structuredContent")
            )

            content = result.get("content", [])

            is_error = (
                result.get("is_error")
                if "is_error" in result
                else result.get("isError", False)
            )

        else:
            structured_content = (
                    getattr(result, "structured_content", None)
                    or getattr(result, "structuredContent", None)
            )

            content = getattr(result, "content", [])

            is_error = getattr(
                result,
                "is_error",
                getattr(result, "isError", False),
            )

        return {
            "success": not bool(is_error),
            "structured_content": structured_content,
            "content": content,
            "is_error": bool(is_error),
        }

    async def list_tools(
            self,
            server_url: str,
            transport: str = "streamable_http",
    ) -> list[dict[str, Any]]:
        """
        获取 MCP Server 暴露的工具列表。
        根据 transport 选择对应的 MCP 传输协议。
        """

        self._validate_url(server_url)

        try:
            async with self._create_transport(
                    server_url,
                    transport,
            ) as (read_stream, write_stream):

                async with ClientSession(
                        read_stream,
                        write_stream,
                ) as session:
                    await session.initialize()

                    result = await session.list_tools()

                    return [
                        self._normalize_tool(tool)
                        for tool in result.tools
                    ]

        except Exception as exc:

            raise RuntimeError(
                f"MCP 工具发现失败，"
                f"Server URL: {server_url}，"
                f"Transport: {transport}，"
                f"原因: {exc}"
            ) from exc

    async def call_tool(
            self,
            server_url: str,
            tool_name: str,
            arguments: dict[str, Any],
            transport: str = "streamable_http",
    ) -> dict[str, Any]:
        """
        调用 MCP Server 上的指定工具。
        """

        self._validate_url(server_url)

        try:
            async with self._create_transport(
                    server_url,
                    transport,
            ) as (read_stream, write_stream):

                async with ClientSession(
                        read_stream,
                        write_stream,
                ) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        tool_name,
                        arguments=arguments,
                    )

                    return self._normalize_result(result)

        except Exception as exc:
            raise RuntimeError(
                f"MCP 工具调用失败，"
                f"Server URL: {server_url}，"
                f"Tool: {tool_name}，"
                f"Transport: {transport}，"
                f"原因: {exc}"
            ) from exc


    def _create_transport(self, server_url: str, transport: str):
        """
        根据 MCP Server 配置选择传输协议。

        sse:
            传统 SSE 协议，连接 /sse 端点。

        streamable_http:
            新版 Streamable HTTP 协议，通常连接 /mcp 端点。
        """

        if transport == "sse":
            return sse_client(server_url)

        if transport == "streamable_http":
            return streamable_http_client(server_url)

        raise ValueError(
            f"不支持的 MCP 传输协议: {transport}"
        )

# 全局 MCP Provider 实例
mcp_provider = MCPProvider()
