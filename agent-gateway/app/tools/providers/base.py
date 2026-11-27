
"""
Tool Provider 基础接口。

所有工具来源都需要实现统一接口。

例如：
- BuiltinProvider：系统内置工具
- MCPProvider：远程 MCP Server
- RemoteProvider：普通 HTTP / RPC 工具服务

这样上层业务不需要关心工具具体来自哪里。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseToolProvider(ABC):
    """所有工具 Provider 的抽象基类。"""

    @abstractmethod
    async def list_tools(
        self,
        server_url: str,
        transport: str = "streamable_http",
    ) -> list[dict[str, Any]]:
        """
        获取远程服务提供的工具列表。

        返回统一格式的工具描述。
        """
        raise NotImplementedError

    @abstractmethod
    async def call_tool(
        self,
        server_url: str,
        tool_name: str,
        arguments: dict[str, Any],
        transport: str = "streamable_http",
    ) -> dict[str, Any]:
        """
        执行指定工具。

        参数：
        server_url：工具服务地址
        tool_name：工具名称
        arguments：工具参数
        """
        raise NotImplementedError
