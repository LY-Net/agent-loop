
from typing import Any, Dict, Optional, Set

from app.planner.schemas import (
    ExecutionPlan,
    ExecutionStep,
    PlanStatus,
    StepStatus,
    StepType,
)

from app.executor.context import ExecutionContext
from app.executor.registry import executor_registry


class ExecutionEngine:
    """
    Agent 执行引擎。

    核心职责：

    1. 接收 ExecutionPlan。
    2. 校验 Step 之间的依赖关系。
    3. 根据依赖关系安排执行顺序。
    4. 根据 StepType 找到对应 Handler。
    5. 执行任务并保存结果。
    6. 处理失败和依赖跳过。
    7. 更新 ExecutionPlan 最终状态。

    注意：
    当前版本采用串行调度。

    后续可以在此基础上扩展：
        - 并行执行
        - 超时控制
        - 重试机制
        - 最大执行时间
        - 人工审批
        - 中断恢复
    """

    def __init__(self):
        self.registry = executor_registry

    # --------------------------------------------------
    # 1. 初始化执行上下文
    # --------------------------------------------------

    def _build_context(
        self,
        plan: ExecutionPlan,
        user_input: str,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        shared_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionContext:
        """
        创建当前 Plan 的执行上下文。
        """

        return ExecutionContext(
            user_input=user_input,
            plan_id=str(plan.plan_id),
            run_id=run_id,
            trace_id=trace_id,
            shared_data=shared_data or {},
        )

    # --------------------------------------------------
    # 2. Step 依赖合法性检查
    # --------------------------------------------------

    def _validate_dependencies(
        self,
        plan: ExecutionPlan,
    ) -> None:
        """
        检查执行计划的依赖关系。

        主要检查：

        1. step_id 是否重复。
        2. depends_on 是否引用不存在的 Step。
        3. 是否依赖自己。
        4. 是否存在循环依赖。
        """

        step_map = {}

        for step in plan.steps:

            step_id = str(step.step_id)

            if step_id in step_map:
                raise ValueError(
                    f"ExecutionPlan 存在重复 step_id: {step_id}"
                )

            step_map[step_id] = step

        # 检查依赖引用是否合法。
        for step in plan.steps:

            current_id = str(step.step_id)

            for dependency_id in step.depends_on:

                dependency_id = str(dependency_id)

                if dependency_id == current_id:
                    raise ValueError(
                        f"Step 不能依赖自己: {current_id}"
                    )

                if dependency_id not in step_map:
                    raise ValueError(
                        f"Step {current_id} 依赖了不存在的 Step: "
                        f"{dependency_id}"
                    )

        # 使用 DFS 检查循环依赖。
        #
        # visiting：当前递归路径中的节点。
        # visited：已经检查完成的节点。
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def dfs(step_id: str):

            if step_id in visiting:
                raise ValueError(
                    f"ExecutionPlan 存在循环依赖: {step_id}"
                )

            if step_id in visited:
                return

            visiting.add(step_id)

            step = step_map[step_id]

            for dependency_id in step.depends_on:
                dfs(str(dependency_id))

            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in step_map:
            dfs(step_id)

    # --------------------------------------------------
    # 3. 更新 Step 状态
    # --------------------------------------------------

    @staticmethod
    def _set_step_status(
        step: ExecutionStep,
        status: StepStatus,
    ) -> None:
        """
        更新 Step 状态。

        使用统一入口，避免在不同执行分支中
        分散修改状态的逻辑。
        """

        step.status = status

    # --------------------------------------------------
    # 4. 判断依赖是否已经全部完成
    # --------------------------------------------------

    @staticmethod
    def _dependencies_completed(
        step: ExecutionStep,
        step_map: Dict[str, ExecutionStep],
    ) -> bool:
        """
        判断当前 Step 的所有前置依赖是否成功。
        """

        for dependency_id in step.depends_on:

            dependency = step_map[str(dependency_id)]

            if dependency.status != StepStatus.SUCCESS:
                return False

        return True

    # --------------------------------------------------
    # 5. 判断是否存在失败依赖
    # --------------------------------------------------

    @staticmethod
    def _has_failed_dependency(
        step: ExecutionStep,
        step_map: Dict[str, ExecutionStep],
    ) -> bool:
        """
        如果某个前置 Step 失败或被跳过，
        那么当前 Step 就不应该继续执行。
        """

        for dependency_id in step.depends_on:

            dependency = step_map[str(dependency_id)]

            if dependency.status in (
                StepStatus.FAILED,
                StepStatus.SKIPPED,
            ):
                return True

        return False

    # --------------------------------------------------
    # 6. 执行单个 Step
    # --------------------------------------------------

    async def _execute_step(
        self,
        step: ExecutionStep,
        context: ExecutionContext,
    ) -> None:
        """
        执行单个 Step。

        执行流程：

            PENDING
                ↓
            RUNNING
                ↓
            SUCCESS / FAILED

        Handler 只负责业务执行，
        Executor 负责统一控制状态和异常。
        """

        step_id = str(step.step_id)

        self._set_step_status(
            step,
            StepStatus.RUNNING,
        )

        try:

            # 根据 StepType 找到对应 Handler。
            handler = self.registry.get_handler(
                step.step_type
            )

            # 执行业务处理器。
            output = await handler(
                step,
                context,
            )

            # 要求所有 Handler 返回字典。
            if not isinstance(output, dict):
                raise TypeError(
                    f"Step Handler 必须返回 dict，"
                    f"当前类型: {type(output).__name__}"
                )

            # 保存 Step 执行结果。
            step.output_data = output

            # 同时保存到共享上下文。
            context.set_step_output(
                step_id,
                output,
            )

            # 执行成功。
            self._set_step_status(
                step,
                StepStatus.SUCCESS,
            )

        except Exception as exc:

            # 保存失败信息。
            error_data = {
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

            step.output_data = error_data

            context.add_error(
                step_id,
                str(exc),
            )

            self._set_step_status(
                step,
                StepStatus.FAILED,
            )

    # --------------------------------------------------
    # 7. 执行计划
    # --------------------------------------------------

    async def execute(
        self,
        plan: ExecutionPlan,
        user_input: str,
        run_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        shared_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """
        执行完整 ExecutionPlan。

        当前使用串行调度策略：

        - 没有依赖的 Step 可以立即执行。
        - 有依赖的 Step 必须等前置 Step 成功。
        - 前置 Step 失败，后续依赖步骤跳过。
        - 不依赖失败步骤的其他任务仍可以执行。
        """

        # --------------------------------------------------
        # A. 澄清计划不进入普通执行阶段
        # --------------------------------------------------

        if plan.need_clarification:

            plan.status = PlanStatus.NEED_CLARIFICATION

            return plan

        # --------------------------------------------------
        # B. 检查执行计划是否合法
        # --------------------------------------------------

        self._validate_dependencies(plan)

        if not plan.steps:
            plan.status = PlanStatus.COMPLETED
            return plan

        # --------------------------------------------------
        # C. 初始化上下文
        # --------------------------------------------------

        context = self._build_context(
            plan=plan,
            user_input=user_input,
            run_id=run_id,
            trace_id=trace_id,
            shared_data=shared_data,
        )

        # 方便通过 step_id 快速定位 Step。
        step_map = {
            str(step.step_id): step
            for step in plan.steps
        }

        # --------------------------------------------------
        # D. 初始化 Step 状态
        # --------------------------------------------------

        for step in plan.steps:

            # 本次执行只接受尚未执行的计划。
            if step.status != StepStatus.PENDING:
                raise ValueError(
                    f"当前计划包含非 PENDING 状态的 Step: "
                    f"{step.step_id}, status={step.status}"
                )

            step.output_data = {}

        plan.status = PlanStatus.RUNNING

        # --------------------------------------------------
        # E. 按依赖关系调度
        # --------------------------------------------------

        while True:

            # 当前仍然等待执行的 Step。
            pending_steps = [
                step
                for step in plan.steps
                if step.status == StepStatus.PENDING
            ]

            # 全部 Step 已经执行完毕。
            if not pending_steps:
                break

            progressed = False

            for step in pending_steps:

                # 如果前置任务已经失败，
                # 当前 Step 无法继续执行。
                if self._has_failed_dependency(
                    step,
                    step_map,
                ):

                    self._set_step_status(
                        step,
                        StepStatus.SKIPPED,
                    )

                    step.output_data = {
                        "reason": "前置依赖执行失败或被跳过"
                    }

                    progressed = True
                    continue

                # 如果前置任务尚未全部成功，
                # 暂时等待下一轮调度。
                if not self._dependencies_completed(
                    step,
                    step_map,
                ):
                    continue

                # 前置任务全部成功，执行当前 Step。
                await self._execute_step(
                    step,
                    context,
                )

                progressed = True

            # 如果这一轮没有任何 Step 可以推进，
            # 说明执行计划存在无法解决的状态。
            #
            # 正常情况下，循环依赖已经在前面被检查，
            # 这里主要用于防御异常状态。
            if not progressed:

                raise RuntimeError(
                    "ExecutionEngine 无法继续调度，"
                    "请检查 Step 状态和依赖关系。"
                )

        # --------------------------------------------------
        # F. 汇总最终执行状态
        # --------------------------------------------------

        failed_steps = [
            step
            for step in plan.steps
            if step.status == StepStatus.FAILED
        ]

        if failed_steps:

            plan.status = PlanStatus.FAILED

        else:

            plan.status = PlanStatus.COMPLETED

        return plan


# 全局执行引擎实例。
execution_engine = ExecutionEngine()
