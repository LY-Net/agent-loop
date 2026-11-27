
"""
Execution Planner

职责：
1. 根据 Intent Engine 的识别结果构建 ExecutionPlan。
2. 根据 execution_mode 决定执行步骤。
3. 将识别出的业务参数传递给 TOOL_CALL。
4. 将 required_capabilities 传递给工具调度器。
5. 建立 RAG、TOOL、LLM 之间的依赖关系。
6. 处理需要用户澄清的情况。

设计原则：
Planner 只描述需要什么能力、需要传递什么参数，
不直接决定具体调用哪个业务工具。

具体工具由 ToolDispatcher 根据 ToolRegistry 动态匹配。
"""

from typing import Any, Dict, List

from app.intent.schemas import (
    IntentResult,
    IntentStatus,
    ExecutionMode,
)

from app.planner.schemas import (
    ExecutionPlan,
    ExecutionStep,
    StepType,
    PlanStatus,
)


class ExecutionPlanner:

    # ========================================================
    # 1. 构建执行计划
    # ========================================================

    def build_plan(
        self,
        intent_result: IntentResult,
        user_input: str,
    ) -> ExecutionPlan:

        # ----------------------------------------------------
        # 1.1 意图识别失败，不能生成执行计划
        # ----------------------------------------------------

        if intent_result.status == IntentStatus.FAILED:
            raise ValueError(
                "Cannot build plan from failed intent result"
            )

        fine = intent_result.fine

        # ----------------------------------------------------
        # 1.2 判断是否需要用户补充信息
        # ----------------------------------------------------

        need_clarification = (
            intent_result.need_clarification
            or (
                fine is not None
                and fine.need_clarification
            )
        )

        if need_clarification:

            question = (
                intent_result.clarification_question
                or (
                    fine.clarification_question
                    if fine is not None
                    else None
                )
                or "请补充必要信息。"
            )

            return ExecutionPlan(
                intent_name=intent_result.final_intent or "unknown",
                execution_mode="clarification",
                description="等待用户补充必要信息",
                status=PlanStatus.NEED_CLARIFICATION,
                need_clarification=True,
                clarification_question=question,
                steps=[
                    ExecutionStep(
                        step_type=StepType.CLARIFY,
                        name="clarify_user_input",
                        description="向用户询问缺失信息",
                        input_data={
                            "question": question,
                        },
                    )
                ],
            )

        # ----------------------------------------------------
        # 1.3 没有 Fine Intent 时，按简单请求处理
        # ----------------------------------------------------

        if fine is None:
            return self._build_direct_plan(
                intent_result.final_intent or "general",
                user_input,
                intent_result.total_confidence,
            )

        # ----------------------------------------------------
        # 1.4 提取 Fine Intent 中的工具调用信息
        # ----------------------------------------------------

        mode = fine.execution_mode

        # Intent Engine 已经识别出的业务参数
        # 例如：
        # {
        #     "region": "重庆地区",
        #     "time_range": "最近一个月",
        #     "metric": "sales_amount"
        # }
        tool_arguments = dict(fine.parameters or {})

        # 工具需要具备的能力
        # 例如：
        # ["sales_query", "database_read"]
        required_capabilities = list(
            fine.required_capabilities or []
        )

        # ----------------------------------------------------
        # 1.5 根据 execution_mode 构建基础步骤
        # ----------------------------------------------------

        if mode == ExecutionMode.DIRECT:

            steps = self._build_direct_steps(
                user_input
            )

        elif mode == ExecutionMode.RAG:

            steps = self._build_rag_steps(
                user_input
            )

        elif mode == ExecutionMode.TOOL:

            steps = self._build_tool_steps(
                user_input,
                required_capabilities,
                tool_arguments,
            )

        elif mode == ExecutionMode.HYBRID:

            steps = self._build_hybrid_steps(
                user_input,
                required_capabilities,
                tool_arguments,
            )

        elif mode == ExecutionMode.PLANNING:

            steps = self._build_planning_steps(
                user_input
            )

        else:
            raise ValueError(
                f"Unsupported execution mode: {mode}"
            )

        # ----------------------------------------------------
        # 1.6 检查当前计划是否已经包含 RAG / TOOL 步骤
        # ----------------------------------------------------

        has_rag = any(
            step.step_type == StepType.RAG_RETRIEVAL
            for step in steps
        )

        has_tool = any(
            step.step_type == StepType.TOOL_CALL
            for step in steps
        )

        rag_step = None
        tool_step = None

        # ----------------------------------------------------
        # 1.7 根据 need_rag 补充 RAG 步骤
        # ----------------------------------------------------

        if fine.need_rag and not has_rag:

            rag_step = ExecutionStep(
                step_type=StepType.RAG_RETRIEVAL,
                name="knowledge_retrieval",
                description="检索相关知识",
                input_data={
                    "query": user_input,
                },
            )

            steps.insert(0, rag_step)

        # ----------------------------------------------------
        # 1.8 根据 need_tool 补充 TOOL 步骤
        # ----------------------------------------------------

        if fine.need_tool and not has_tool:

            tool_step = self._build_tool_step(
                user_input=user_input,
                capabilities=required_capabilities,
                parameters=tool_arguments,
            )

            # 如果前面插入了 RAG，则让 TOOL 排在 RAG 后面
            if rag_step is not None:
                steps.insert(1, tool_step)
            else:
                steps.insert(0, tool_step)

        # ----------------------------------------------------
        # 1.9 重新建立步骤依赖关系
        #
        # 统一依赖顺序：
        # RAG -> TOOL -> LLM_GENERATION
        # ----------------------------------------------------

        rag_id = next(
            (
                step.step_id
                for step in steps
                if step.step_type == StepType.RAG_RETRIEVAL
            ),
            None,
        )

        tool_id = next(
            (
                step.step_id
                for step in steps
                if step.step_type == StepType.TOOL_CALL
            ),
            None,
        )

        for step in steps:

            # TOOL 依赖 RAG
            if step.step_type == StepType.TOOL_CALL and rag_id:

                if rag_id not in step.depends_on:
                    step.depends_on.append(rag_id)

            # LLM 同时依赖 RAG 和 TOOL
            elif step.step_type == StepType.LLM_GENERATION:

                if rag_id and rag_id not in step.depends_on:
                    step.depends_on.append(rag_id)

                if tool_id and tool_id not in step.depends_on:
                    step.depends_on.append(tool_id)

        # ----------------------------------------------------
        # 1.10 生成最终 ExecutionPlan
        # ----------------------------------------------------

        return ExecutionPlan(
            intent_name=fine.primary_intent,
            execution_mode=mode.value,
            description=fine.intent_description,
            steps=steps,
            metadata={
                "confidence": fine.confidence,
                "required_capabilities": required_capabilities,
                "need_rag": fine.need_rag,
                "need_tool": fine.need_tool,
                "tool_arguments": tool_arguments,
            },
        )

    # ========================================================
    # 2. 统一构建 TOOL_CALL 输入
    # ========================================================

    def _build_tool_input(
        self,
        user_input: str,
        capabilities: List[str],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        构建统一的工具调用输入结构。

        注意：
        这里不再写死 tool_name。

        Dispatcher 后续根据 required_capabilities
        自动匹配 ToolRegistry 中的具体工具。
        """

        return {
            "query": user_input,

            # Intent Engine 提取的结构化参数
            "arguments": parameters,

            # 工具能力要求
            "required_capabilities": capabilities,
        }

    # ========================================================
    # 3. 构建单个 TOOL_CALL Step
    # ========================================================

    def _build_tool_step(
        self,
        user_input: str,
        capabilities: List[str],
        parameters: Dict[str, Any],
    ) -> ExecutionStep:

        return ExecutionStep(
            step_type=StepType.TOOL_CALL,
            name="execute_tool",
            description="根据所需能力动态匹配并执行工具",
            input_data=self._build_tool_input(
                user_input,
                capabilities,
                parameters,
            ),
            required_capabilities=capabilities,
        )

    # ========================================================
    # 4. DIRECT 模式
    # ========================================================

    def _build_direct_plan(
        self,
        intent_name: str,
        user_input: str,
        confidence: float,
    ) -> ExecutionPlan:

        return ExecutionPlan(
            intent_name=intent_name,
            execution_mode="direct",
            description="直接处理用户请求",
            steps=self._build_direct_steps(user_input),
            metadata={
                "confidence": confidence,
            },
        )

    def _build_direct_steps(
        self,
        user_input: str,
    ) -> List[ExecutionStep]:

        return [
            ExecutionStep(
                step_type=StepType.LLM_GENERATION,
                name="generate_response",
                description="直接生成回答",
                input_data={
                    "user_input": user_input,
                },
            )
        ]

    # ========================================================
    # 5. RAG 模式
    # ========================================================

    def _build_rag_steps(
        self,
        user_input: str,
    ) -> List[ExecutionStep]:

        rag = ExecutionStep(
            step_type=StepType.RAG_RETRIEVAL,
            name="knowledge_retrieval",
            description="检索相关知识",
            input_data={
                "query": user_input,
            },
        )

        generation = ExecutionStep(
            step_type=StepType.LLM_GENERATION,
            name="generate_response",
            description="结合检索结果生成回答",
            depends_on=[rag.step_id],
            input_data={
                "user_input": user_input,
            },
        )

        return [rag, generation]

    # ========================================================
    # 6. TOOL 模式
    # ========================================================

    def _build_tool_steps(
        self,
        user_input: str,
        capabilities: List[str],
        parameters: Dict[str, Any],
    ) -> List[ExecutionStep]:

        tool = self._build_tool_step(
            user_input,
            capabilities,
            parameters,
        )

        generation = ExecutionStep(
            step_type=StepType.LLM_GENERATION,
            name="generate_response",
            description="根据工具执行结果生成回答",
            depends_on=[tool.step_id],
            input_data={
                "user_input": user_input,
            },
        )

        return [tool, generation]

    # ========================================================
    # 7. HYBRID 模式
    # ========================================================

    def _build_hybrid_steps(
        self,
        user_input: str,
        capabilities: List[str],
        parameters: Dict[str, Any],
    ) -> List[ExecutionStep]:

        rag = ExecutionStep(
            step_type=StepType.RAG_RETRIEVAL,
            name="knowledge_retrieval",
            description="检索知识库",
            input_data={
                "query": user_input,
            },
        )

        tool = self._build_tool_step(
            user_input,
            capabilities,
            parameters,
        )

        # HYBRID 模式下，先检索，再调用工具
        tool.depends_on = [rag.step_id]

        generation = ExecutionStep(
            step_type=StepType.LLM_GENERATION,
            name="generate_response",
            description="综合检索和工具结果生成回答",
            depends_on=[
                rag.step_id,
                tool.step_id,
            ],
            input_data={
                "user_input": user_input,
            },
        )

        return [rag, tool, generation]

    # ========================================================
    # 8. PLANNING 模式
    # ========================================================

    def _build_planning_steps(
        self,
        user_input: str,
    ) -> List[ExecutionStep]:

        planning = ExecutionStep(
            step_type=StepType.TASK_PLANNING,
            name="decompose_task",
            description="将复杂任务拆解为可执行子任务",
            input_data={
                "user_input": user_input,
            },
        )

        generation = ExecutionStep(
            step_type=StepType.LLM_GENERATION,
            name="generate_response",
            description="根据任务执行结果生成最终回答",
            depends_on=[planning.step_id],
            input_data={
                "user_input": user_input,
            },
        )

        return [planning, generation]


# ============================================================
# 全局 Planner 单例
# ============================================================

execution_planner = ExecutionPlanner()
