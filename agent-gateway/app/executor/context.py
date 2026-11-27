
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ExecutionContext:
    """
    Agent 执行上下文。

    一个 ExecutionContext 对应一次完整的 Agent 执行过程。

    不同 Step 之间可以通过该对象共享数据，
    但每个 Step 的正式输出仍然保存在自己的 output_data 中。
    """

    # 当前用户输入。
    # 例如：帮我分析一下最近一个月的销售数据。
    user_input: str

    # 当前执行计划 ID。
    plan_id: Optional[str] = None

    # 当前 Agent Run ID。
    # 用于后续关联 Trace、Run、Span、Event。
    run_id: Optional[str] = None

    # 当前 Trace ID。
    trace_id: Optional[str] = None

    # 保存每个 Step 的执行结果。
    #
    # 数据结构：
    # {
    #     "step_id_1": {"result": "..."},
    #     "step_id_2": {"result": "..."}
    # }
    step_outputs: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )

    # 执行过程中共享的其他数据。
    #
    # 例如：
    # {
    #     "user_id": "user_001",
    #     "session_id": "session_001"
    # }
    shared_data: Dict[str, Any] = field(
        default_factory=dict
    )

    # 当前执行过程中的错误信息。
    errors: list = field(
        default_factory=list
    )

    def set_step_output(
        self,
        step_id: str,
        output: Dict[str, Any],
    ) -> None:
        """
        保存某个 Step 的执行结果。
        """

        self.step_outputs[step_id] = output

    def get_step_output(
        self,
        step_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        获取指定 Step 的执行结果。
        """

        return self.step_outputs.get(step_id)

    def add_error(
        self,
        step_id: str,
        error: str,
    ) -> None:
        """
        记录 Step 执行错误。
        """

        self.errors.append({
            "step_id": step_id,
            "error": error,
        })
