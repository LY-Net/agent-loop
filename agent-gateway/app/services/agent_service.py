
import uuid
import time

from datetime import datetime, timezone

from sqlalchemy import update

from app.db.session import AsyncSessionLocal

from app.models.observability import (
    AgentTrace,
    AgentRun,
)

from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
)

from app.intent.engine import intent_engine
from app.intent.schemas import IntentStatus

from app.planner.planner import execution_planner
from app.planner.validation import execution_plan_validator

from app.observability.context import (
    set_execution_context,
    reset_execution_context,
)

from app.observability.recorder import (
    SpanRecorder,
    record_event,
)

from app.repositories.agent_session_repository import (
    AgentSessionRepository,
)

from app.services.clarification_manager import (
    ClarificationManager,
)

def utc_now():
    return datetime.now(timezone.utc)


class AgentService:
    def __init__(self):
        self.clarification_manager = ClarificationManager()

    async def chat(
        self,
        request: AgentChatRequest,
        request_id: str,
        trace_id: str,
    ) -> AgentChatResponse:

        session_id = request.session_id or str(uuid.uuid4())
        run_id = str(uuid.uuid4())

        start_clock = time.perf_counter()

        # 设置当前请求的 Trace / Run 上下文
        tokens = set_execution_context(
            trace_id,
            run_id,
        )

        trace_status = "success"
        error_message = None
        result = {}
        records_created = False

        try:
            # --------------------------------------------------
            # 1. 创建 Trace 和 Run
            # --------------------------------------------------
            async with AsyncSessionLocal() as session:

                trace = AgentTrace(
                    trace_id=trace_id,
                    request_id=request_id,
                    user_id=request.user_id,
                    session_id=session_id,
                    status="running",
                    input_data={
                        "message": request.message,
                        "metadata": request.metadata,
                    },
                )

                run = AgentRun(
                    run_id=run_id,
                    trace_id=trace_id,
                    agent_name="default_agent",
                    status="running",
                    input_data={
                        "message": request.message,
                    },
                )

                # 先写入 Trace
                session.add(trace)
                await session.flush()

                print(
                    "TRACE INSERTED:",
                    trace.trace_id,
                    flush=True,
                )

                # 再写入 Run
                session.add(run)
                await session.flush()

                print(
                    "RUN INSERTED:",
                    run.run_id,
                    flush=True,
                )

                # 最后统一提交
                await session.commit()

                records_created = True

                print(
                    "TRACE AND RUN COMMITTED",
                    flush=True,
                )

            # --------------------------------------------------
            # 2. 开始 Agent Loop
            # --------------------------------------------------
            async with SpanRecorder(
                "agent_loop",
                "agent",
                {
                    "message": request.message,
                },
            ) as root_span:

                # --------------------------------------------------
                # 3. 调用真实 Intent Engine
                # --------------------------------------------------
                # --------------------------------------------------
                # 3. 恢复会话上下文
                # --------------------------------------------------

                current_input = request.message
                session_context = None
                pending_clarification = None
                is_clarification_reply = False

                if request.session_id:

                    async with AsyncSessionLocal() as session:

                        repository = AgentSessionRepository(session)

                        agent_session = await repository.get_by_session_id(
                            session_id=session_id,
                            user_id=request.user_id,
                        )

                        if agent_session is not None:

                            if agent_session.status == "waiting_clarification":
                                session_context = agent_session.context_data
                                pending_clarification = (
                                    agent_session.pending_clarification
                                )

                                is_clarification_reply = True

                # --------------------------------------------------
                # 4. 合并用户补充信息
                # --------------------------------------------------

                if is_clarification_reply:
                    clarification_result = (
                        self.clarification_manager.apply_user_response(
                            context_data=session_context,
                            pending_clarification=pending_clarification,
                            user_response=request.message,
                        )
                    )

                    session_context = clarification_result["context_data"]
                    current_input = clarification_result["combined_input"]

                # --------------------------------------------------
                # 5. 调用 Intent Engine
                # --------------------------------------------------

                intent_result = await intent_engine.recognize(
                    current_input,
                )

                # 转成 JSON 兼容的数据，方便存储到 JSON 字段
                intent_data = intent_result.model_dump(
                    mode="json",
                )

                # 意图识别失败时，终止本次 Agent 执行
                if intent_result.status == IntentStatus.FAILED:
                    raise RuntimeError(
                        "Intent recognition failed: "
                        f"{intent_data}"
                    )

                # --------------------------------------------------
                # 保存或更新 AgentSession
                # --------------------------------------------------

                fine = intent_result.fine

                need_clarification = intent_result.need_clarification

                if need_clarification:
                    response_message = (
                            intent_result.clarification_question
                            or (
                                fine.clarification_question
                                if fine is not None else None
                            )
                            or "需要进一步确认您的需求。"
                    )
                else:
                    response_message = "Agent 意图识别执行成功"

                if need_clarification:

                    # 第一次请求，构建初始上下文
                    if session_context is None:
                        session_context = (
                            self.clarification_manager.build_initial_context(
                                original_message=request.message,
                                intent_name=intent_result.final_intent,
                                known_parameters=(
                                    fine.parameters if fine is not None else {}
                                ),
                                required_capabilities=(
                                    fine.required_capabilities
                                    if fine is not None else []
                                ),
                            )
                        )

                    # 构建待澄清问题
                    pending_data = (
                        self.clarification_manager.build_pending_clarification(
                            question=(
                                    intent_result.clarification_question
                                    or (
                                        fine.clarification_question
                                        if fine is not None else None
                                    )
                                    or "请补充更多信息，以便我准确理解您的需求。"
                            ),
                            known_parameters=(
                                fine.parameters if fine is not None else {}
                            ),
                        )
                    )

                    async with AsyncSessionLocal() as session:

                        repository = AgentSessionRepository(session)

                        existing = await repository.get_by_session_id(
                            session_id=session_id,
                            user_id=request.user_id,
                        )

                        if existing is None:

                            await repository.create_session(
                                user_id=request.user_id,
                                session_id=session_id,
                                context_data=session_context,
                            )

                            # 创建后再保存澄清状态
                            await repository.save_clarification(
                                session_id=session_id,
                                user_id=request.user_id,
                                context_data=session_context,
                                pending_clarification=pending_data,
                            )

                        else:

                            await repository.save_clarification(
                                session_id=session_id,
                                user_id=request.user_id,
                                context_data=session_context,
                                pending_clarification=pending_data,
                            )

                elif is_clarification_reply:

                    # 用户补充后，意图识别已满足要求
                    async with AsyncSessionLocal() as session:

                        repository = AgentSessionRepository(session)

                        await repository.complete_session(
                            session_id=session_id,
                            user_id=request.user_id,
                            context_data=session_context,
                        )


                # --------------------------------------------------
                # 4. 根据 Intent Result 生成 Execution Plan
                # --------------------------------------------------

                need_rag = (
                        intent_result.fine is not None
                        and intent_result.fine.need_rag
                )

                execution_plan = execution_planner.build_plan(
                    intent_result,
                    current_input,
                )

                # --------------------------------------------------
                # 5. 校验 Execution Plan
                # --------------------------------------------------

                # ===== Planner / Validator 数据一致性检查 =====
                fine = intent_result.fine

                print("\n========== PLAN CONTRACT DEBUG ==========")

                print("Intent status:", intent_result.status)
                print("Intent final:", intent_result.final_intent)
                print("Intent need_clarification:", intent_result.need_clarification)

                if fine is None:
                    print("Intent fine: None")
                else:
                    print("Fine primary_intent:", fine.primary_intent)
                    print("Fine need_tool:", fine.need_tool)
                    print("Fine need_rag:", fine.need_rag)
                    print("Fine execution_mode:", fine.execution_mode)

                print("\nExecution Plan:")
                print("Plan intent:", execution_plan.intent_name)
                print("Plan mode:", execution_plan.execution_mode)

                for index, step in enumerate(execution_plan.steps):
                    print(
                        f"Step {index + 1}:",
                        "type =", step.step_type,
                        "| type_value =", getattr(step.step_type, "value", step.step_type),
                        "| name =", step.name,
                    )

                print("========================================\n")
                execution_plan_validator.validate(
                    intent_result,
                    execution_plan,
                )

                # 转换为 JSON 兼容的数据
                plan_data = execution_plan.model_dump(
                    mode="json",
                )
                # 从执行计划中判断是否包含 RAG 步骤
                need_rag = any(
                    getattr(step.step_type, "value", step.step_type)
                    == "rag_retrieval"
                    for step in execution_plan.steps
                )
                # --------------------------------------------------
                # 6. 记录计划生成事件
                # --------------------------------------------------
                await record_event(
                    "agent.plan.created",
                    {
                        "plan_id": str(execution_plan.plan_id),
                        "intent": intent_result.final_intent,
                        "execution_mode": str(
                            getattr(
                                execution_plan.execution_mode,
                                "value",
                                execution_plan.execution_mode,
                            )
                        ),
                        "step_count": len(execution_plan.steps),
                        "status": str(
                            getattr(
                                execution_plan.status,
                                "value",
                                execution_plan.status,
                            )
                        ),
                    },
                )

                # --------------------------------------------------
                # 7. 汇总 Agent 当前执行结果
                # --------------------------------------------------
                result = {
                    "status": "success",
                    "intent": intent_data,
                    "execution_plan": plan_data,
                    "execution_status": "planned_not_executed",
                }

                # 注意：set_output 是同步方法，不能 await
                root_span.set_output(result)


            # --------------------------------------------------
            # 6. 记录 Agent 执行完成事件
            # --------------------------------------------------
            await record_event(
                "agent.run.completed",
                {
                    "status": "success",
                    "intent": intent_result.final_intent,
                    "need_rag": need_rag,
                },
            )

            # --------------------------------------------------
            # 7. 组织 API 返回信息
            # --------------------------------------------------
            if intent_result.need_clarification:
                response_message = (
                    intent_result.clarification_question
                    or "需要进一步确认您的需求。"
                )
            else:
                response_message = "Agent 意图识别执行成功"

        except Exception as exc:

            import traceback

            trace_status = "error"
            error_message = str(exc)

            print(
                "\n========== ORIGINAL AGENT ERROR =========="
            )
            traceback.print_exc()
            print(
                "==========================================\n"
            )

            # 暂时不在这里再次调用 record_event，
            # 避免记录失败时产生二次异常，覆盖原始错误。
            raise

        finally:

            duration_ms = int(
                (time.perf_counter() - start_clock) * 1000
            )

            try:
                # 只有 Trace / Run 成功创建并提交后，
                # 才更新它们的最终状态。
                if records_created:

                    async with AsyncSessionLocal() as session:

                        await session.execute(
                            update(AgentRun)
                            .where(
                                AgentRun.run_id == run_id
                            )
                            .values(
                                status=trace_status,
                                output_data=(
                                    result
                                    if trace_status == "success"
                                    else None
                                ),
                                error_message=error_message,
                                ended_at=utc_now(),
                                duration_ms=duration_ms,
                            )
                        )

                        await session.execute(
                            update(AgentTrace)
                            .where(
                                AgentTrace.trace_id == trace_id
                            )
                            .values(
                                status=trace_status,
                                output_data=(
                                    result
                                    if trace_status == "success"
                                    else None
                                ),
                                error_message=error_message,
                                ended_at=utc_now(),
                                duration_ms=duration_ms,
                            )
                        )

                        await session.commit()

            finally:
                # 无论数据库状态更新是否成功，
                # 都恢复当前请求的 ContextVar。
                reset_execution_context(tokens)

        return AgentChatResponse(
            success=True,
            request_id=request_id,
            trace_id=trace_id,
            session_id=session_id,
            message=response_message,
            data=result,
        )


agent_service = AgentService()
