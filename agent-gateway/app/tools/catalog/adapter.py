
from typing import Any, Dict, Type
from pydantic import BaseModel, Field, create_model, ConfigDict

from app.tools.schemas import ToolDefinition
from app.tools.registry import ToolRegistry
from app.tools.catalog.service import ToolCatalogService


class MCPToolAdapter:
    """
    动态 MCP 工具适配器。

    职责：
    1. 从 Tool Catalog 加载工具定义
    2. 根据 JSON Schema 动态创建 Pydantic 参数模型
    3. 创建通用 MCP Handler
    4. 将工具注册到 ToolRegistry

    注意：
    不为每一个 MCP 工具编写独立的 Python Handler。
    """

    def __init__(
        self,
        catalog_service: ToolCatalogService,
        tool_registry: ToolRegistry,
        mcp_provider: Any,
    ):
        self.catalog_service = catalog_service
        self.tool_registry = tool_registry
        self.mcp_provider = mcp_provider

    # =====================================================
    # 1. JSON Schema -> Python 类型
    # =====================================================

    @staticmethod
    def _resolve_python_type(schema: Dict[str, Any]) -> Type:
        """
        将 JSON Schema 的基础类型转换成 Python 类型。

        第一版支持：
        - string
        - integer
        - number
        - boolean
        - object
        - array
        """

        schema_type = schema.get("type", "string")

        if isinstance(schema_type, list):
            schema_type = next(
                (item for item in schema_type if item != "null"),
                "string",
            )

        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }

        return type_mapping.get(schema_type, Any)

    # =====================================================
    # 2. JSON Schema -> Pydantic Model
    # =====================================================

    @classmethod
    def _build_input_model(
        cls,
        tool_name: str,
        input_schema: Dict[str, Any],
    ) -> Type[BaseModel]:
        """
        根据 MCP 工具的 input_schema 动态创建 Pydantic Model。

        例如 paperSearch：

        {
            "type": "object",
            "required": ["message", "page", "size"],
            "properties": {
                "message": {"type": "string"},
                "page": {"type": "integer"},
                "size": {"type": "integer"}
            }
        }

        将自动生成对应的参数模型。
        """

        properties = input_schema.get("properties", {})
        required_fields = set(
            input_schema.get("required", [])
        )

        fields = {}

        for field_name, field_schema in properties.items():

            python_type = cls._resolve_python_type(
                field_schema
            )

            description = field_schema.get(
                "description",
                "",
            )

            if field_name in required_fields:
                default_value = ...
            else:
                default_value = field_schema.get(
                    "default",
                    None,
                )

            fields[field_name] = (
                python_type,
                Field(
                    default=default_value,
                    description=description,
                ),
            )

        model_name = (
            f"MCP_{tool_name}_Input"
            .replace("-", "_")
            .replace(".", "_")
        )

        return create_model(
            model_name,
            __config__=ConfigDict(
                extra="forbid"
            ),
            **fields,
        )

    # =====================================================
    # 3. 构建通用 MCP Handler
    # =====================================================

    def _build_handler(
        self,
        server_url: str,
        transport: str,
        tool_name: str,
    ):
        """
        创建通用 MCP Handler。

        Handler 不关心具体业务。
        它只负责将参数转发给对应的 MCP Server。
        """

        async def handler(
            input_data: BaseModel,
            context: Any,
        ) -> Dict[str, Any]:

            arguments = input_data.model_dump(
                exclude_none=True
            )

            result = await self.mcp_provider.call_tool(
                server_url=server_url,
                tool_name=tool_name,
                arguments=arguments,
                transport=transport,
            )

            return result

        return handler

    # =====================================================
    # 4. 动态加载并注册 MCP 工具
    # =====================================================

    async def load_tool(self, tool_id: str) -> ToolDefinition:
        """
        根据 Catalog 中的 tool_id 加载 MCP 工具，
        动态转换成 ToolDefinition 并注册。

        不需要硬编码工具名称。
        """

        # -------------------------------------------------
        # A. 从 Catalog 加载工具详情
        # -------------------------------------------------

        tool = await self.catalog_service.load_tool(
            tool_id
        )

        if not tool:
            raise ValueError(
                f"Catalog 工具不存在或不可用: {tool_id}"
            )

        if tool.provider_type != "mcp":
            raise ValueError(
                f"当前适配器只支持 MCP 工具: {tool.provider_type}"
            )

        if not tool.server_id:
            raise ValueError(
                f"MCP 工具缺少 server_id: {tool.name}"
            )

        # -------------------------------------------------
        # B. 获取所属 MCP Server
        # -------------------------------------------------

        server = await self.catalog_service.get_server(
            tool.server_id
        )

        if not server:
            raise ValueError(
                f"MCP Server 不存在: {tool.server_id}"
            )

        if not server.server_url:
            raise ValueError(
                f"MCP Server URL 为空: {server.name}"
            )

        # -------------------------------------------------
        # C. 获取传输协议
        # -------------------------------------------------

        connection_config = (
            server.connection_config or {}
        )

        transport = connection_config.get(
            "transport",
            "streamable_http",
        )

        # -------------------------------------------------
        # D. 动态构建参数模型
        # -------------------------------------------------

        input_model = self._build_input_model(
            tool.name,
            tool.input_schema or {},
        )

        # -------------------------------------------------
        # E. 创建通用 Handler
        # -------------------------------------------------

        handler = self._build_handler(
            server_url=str(server.server_url),
            transport=transport,
            tool_name=tool.name,
        )

        # -------------------------------------------------
        # F. 构建运行时 ToolDefinition
        # -------------------------------------------------

        registry_name = self._build_registry_name(
            tool.server_id,
            tool.name,
        )

        definition = ToolDefinition(
            name=registry_name,
            description=tool.description or "",
            input_model=input_model,
            handler=handler,
            capabilities=tool.capabilities or [],
            requires_confirmation=False,
            metadata={
                "provider_type": "mcp",
                "tool_id": tool.tool_id,
                "server_id": tool.server_id,
                "server_name": server.name,
                "remote_tool_name": tool.name,
                "transport": transport,
                "source_key": getattr(
                    tool,
                    "source_key",
                    None,
                ),
            },
        )

        # 重复加载时先注销旧定义，避免重复注册
        if self.tool_registry.has(registry_name):
            self.tool_registry.unregister(registry_name)

        self.tool_registry.register(definition)

        return definition

        # -------------------------------------------------
        # G. 注册到运行时 ToolRegistry
        # -------------------------------------------------

        self.tool_registry.register(definition)

        return definition

    @staticmethod
    def _build_registry_name(
        server_id: str,
        tool_name: str,
    ) -> str:
        """
        生成 Gateway 内部唯一工具名。

        避免不同 MCP Server 存在同名工具时发生冲突。
        """

        server_prefix = server_id.replace("-", "")[:8]

        safe_tool_name = (
            tool_name
            .replace("-", "_")
            .replace(".", "_")
        )

        return f"mcp_{server_prefix}_{safe_tool_name}"