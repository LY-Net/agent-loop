
"""
Tool Dispatcher

职责：
1. 接收 ExecutionEngine 传入的 TOOL_CALL Step。
2. 根据 tool_name 或 required_capabilities 定位工具。
3. 处理工具匹配歧义。
4. 检查工具执行确认要求。
5. 校验工具参数。
6. 调用具体业务 Handler。
7. 返回统一结果。
"""

from typing import Any, Dict

from app.tools.registry import ToolRegistry, tool_registry


class ToolDispatcher:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    # ========================================================
    # 1. 动态解析具体工具
    # ========================================================

    def _resolve_tool(self, step: Any):
        """
        优先使用显式 tool_name。

        如果没有指定 tool_name，则使用 required_capabilities
        从 ToolRegistry 中匹配工具。

        匹配规则：
            0 个：报错，说明没有可用工具。
            1 个：使用该工具。
            多个：报错，避免错误选择。
        """

        input_data = step.input_data or {}

        # 兼容显式指定工具名称的旧调用方式
        tool_name = input_data.get("tool_name")

        if tool_name:
            tool = self.registry.get(tool_name)

            if tool is None:
                raise ValueError(
                    f"找不到已注册的工具：{tool_name}"
                )

            return tool

        # 从 Step 或 input_data 获取能力要求
        required_capabilities = (
            input_data.get("required_capabilities")
            or getattr(step, "required_capabilities", [])
            or []
        )

        if not required_capabilities:
            raise ValueError(
                "没有指定 tool_name，也没有指定 required_capabilities"
            )

        # 根据能力标签匹配工具
        matched_tools = self.registry.find_by_capabilities(
            required_capabilities
        )

        if not matched_tools:
            raise ValueError(
                "没有找到满足能力要求的工具："
                f"{required_capabilities}"
            )

        if len(matched_tools) > 1:
            matched_names = [
                tool.name for tool in matched_tools
            ]

            raise ValueError(
                "工具匹配存在歧义，匹配到多个工具："
                f"{matched_names}"
            )

        return matched_tools[0]

    # ========================================================
    # 2. 解析工具参数
    # ========================================================

    def _resolve_arguments(
        self,
        step: Any,
    ) -> Dict[str, Any]:

        input_data = step.input_data or {}

        arguments = input_data.get("arguments")

        if not isinstance(arguments, dict):
            raise ValueError(
                "TOOL_CALL 的 arguments 必须是字典"
            )

        return arguments

    # ========================================================
    # 3. 检查工具执行确认
    # ========================================================

    def _check_confirmation(
        self,
        tool: Any,
        context: Any,
    ) -> None:

        if not tool.requires_confirmation:
            return

        shared_data = (
            getattr(context, "shared_data", {}) or {}
        )

        confirmed_tools = shared_data.get(
            "confirmed_tools",
            [],
        )

        if not isinstance(
            confirmed_tools,
            (list, set, tuple),
        ):
            raise PermissionError(
                "工具确认信息格式不正确"
            )

        if tool.name not in confirmed_tools:
            raise PermissionError(
                f"工具 {tool.name} 需要用户确认后才能执行"
            )

    # ========================================================
    # 4. 执行工具
    # ========================================================

    async def dispatch(
        self,
        step: Any,
        context: Any,
    ) -> Dict[str, Any]:

        # 1. 动态匹配工具
        tool = self._resolve_tool(step)

        # 2. 检查确认要求
        self._check_confirmation(
            tool,
            context,
        )

        # 3. 提取原始参数
        raw_arguments = self._resolve_arguments(step)

        # 4. 使用工具对应的 Pydantic 模型校验参数
        validated_arguments = tool.validate_input(
            raw_arguments
        )

        # 5. 调用具体业务 Handler
        result = await tool.handler(
            validated_arguments,
            context,
        )

        # 6. 检查工具返回类型
        if not isinstance(result, dict):
            raise TypeError(
                f"工具 {tool.name} 返回值必须是字典"
            )

        # 7. 统一返回结构
        return {
            "tool_name": tool.name,
            "success": True,
            "result": result,
        }

    # ========================================================
    # 5. 适配 ExecutorRegistry
    # ========================================================

    async def handle_step(
        self,
        step: Any,
        context: Any,
    ) -> Dict[str, Any]:

        return await self.dispatch(
            step,
            context,
        )


# ============================================================
# 全局 Dispatcher 单例
# ============================================================

tool_dispatcher = ToolDispatcher(
    registry=tool_registry
)
