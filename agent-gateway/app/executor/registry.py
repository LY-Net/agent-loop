
from typing import Any, Awaitable, Callable, Dict

from app.planner.schemas import StepType


# StepHandler 是统一的执行器函数类型。
#
# 参数：
#   step：当前需要执行的 ExecutionStep
#   context：当前 Agent 执行上下文
#
# 返回：
#   Dict[str, Any]：当前 Step 的执行结果
#
# 所有处理器都必须遵循这个接口。
StepHandler = Callable[
    [Any, Any],
    Awaitable[Dict[str, Any]]
]


class ExecutorRegistry:
    """
    Step 执行处理器注册中心。

    负责维护：
        StepType -> Handler

    例如：
        TOOL_CALL -> ToolHandler.execute
        RAG_RETRIEVAL -> RagHandler.execute

    Executor 不需要知道具体处理器内部如何实现，
    只需要通过 Registry 获取对应的 Handler。
    """

    def __init__(self):
        self._handlers: Dict[str, StepHandler] = {}

    @staticmethod
    def _normalize_step_type(step_type: Any) -> str:
        """
        统一处理 Enum 和字符串形式的 StepType。

        兼容：
            StepType.TOOL_CALL
            "tool_call"
        """

        return str(
            getattr(
                step_type,
                "value",
                step_type,
            )
        )

    def register(
        self,
        step_type: StepType,
        handler: StepHandler,
    ) -> None:
        """
        注册某种 StepType 对应的处理器。

        如果已经注册相同类型，则覆盖原来的处理器。
        """

        key = self._normalize_step_type(step_type)

        if not callable(handler):
            raise TypeError(
                f"Handler 必须是可调用对象: {key}"
            )

        self._handlers[key] = handler

    def unregister(
        self,
        step_type: StepType,
    ) -> None:
        """
        注销指定类型的处理器。
        """

        key = self._normalize_step_type(step_type)

        self._handlers.pop(
            key,
            None,
        )

    def get_handler(
        self,
        step_type: StepType,
    ) -> StepHandler:
        """
        获取指定类型的处理器。

        如果没有注册，则抛出明确异常。
        """

        key = self._normalize_step_type(step_type)

        handler = self._handlers.get(key)

        if handler is None:
            raise ValueError(
                f"没有注册对应的 Step Handler: {key}"
            )

        return handler

    def has_handler(
        self,
        step_type: StepType,
    ) -> bool:
        """
        判断某种 StepType 是否已经注册。
        """

        key = self._normalize_step_type(step_type)

        return key in self._handlers

    def list_handlers(self) -> list:
        """
        获取当前已注册的处理器类型。
        """

        return list(
            self._handlers.keys()
        )


# 全局 Registry 单例。
#
# 后续在系统初始化时注册：
# executor_registry.register(...)
#
# Agent 执行时直接使用这个实例。
executor_registry = ExecutorRegistry()
