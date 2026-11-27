
import logging
from app.observability.recorder import (
    SpanRecorder,
    record_event,
)
from app.intent.schemas import (
    IntentCategory,
    IntentStatus,
    IntentResult,
)
from app.intent.coarse import (
    CoarseIntentRecognizer,
    coarse_intent_recognizer,
)
from app.intent.fine import (
    FineIntentRecognizer,
    fine_intent_recognizer,
)


logger = logging.getLogger(__name__)


class IntentEngine:
    """统一意图识别引擎。"""

    # 低于该阈值时，即使模型建议跳过精细识别，也继续分析
    COARSE_CONFIDENCE_THRESHOLD = 0.75

    # 低于该阈值时，最终结果标记为 UNCERTAIN
    FINAL_CONFIDENCE_THRESHOLD = 0.60

    # 这些类别在满足条件时允许跳过精细识别
    SIMPLE_CATEGORIES = {
        IntentCategory.GENERAL,
        IntentCategory.KNOWLEDGE_QA,
    }

    def __init__(
        self,
        coarse_recognizer: CoarseIntentRecognizer,
        fine_recognizer: FineIntentRecognizer,
    ):
        self.coarse_recognizer = coarse_recognizer
        self.fine_recognizer = fine_recognizer

    def _should_run_fine(self, coarse_result) -> bool:
        """通过代码规则判断是否需要精细识别。"""

        # 模型明确要求进一步识别
        if coarse_result.need_fine:
            return True

        # 置信度过低，不能直接采用粗粒度结果
        if (
            coarse_result.confidence
            < self.COARSE_CONFIDENCE_THRESHOLD
        ):
            return True

        # 复杂业务类别不允许直接跳过精细识别
        if coarse_result.category not in self.SIMPLE_CATEGORIES:
            return True

        return False

    async def _safe_record_event(
            self,
            event_type: str,
            event_data: dict,
    ) -> None:
        """
        安全记录事件。

        事件记录失败不能覆盖原始业务异常。
        """
        try:
            await record_event(
                event_type=event_type,
                event_data=event_data,
            )
        except Exception:
            logger.exception(
                "Intent Engine 事件记录失败: %s",
                event_type,
            )

    @staticmethod
    def _build_simple_result(coarse_result) -> IntentResult:
        """为简单请求构建无需精细识别的结果。"""

        final_intent = coarse_result.category.value

        if coarse_result.candidates:
            final_intent = coarse_result.candidates[0].intent_name

        status = IntentStatus.SUCCESS

        if (
            coarse_result.confidence
            < IntentEngine.FINAL_CONFIDENCE_THRESHOLD
        ):
            status = IntentStatus.UNCERTAIN

        return IntentResult(
            status=status,
            coarse=coarse_result,
            fine=None,
            final_intent=final_intent,
            total_confidence=coarse_result.confidence,
            need_clarification=False,
            clarification_question=None,
        )

    @staticmethod
    def _build_final_result(
        coarse_result,
        fine_result,
    ) -> IntentResult:
        """整合粗粒度与精细识别结果。"""

        confidence = fine_result.confidence

        # 两阶段结果都参与最终置信度评估。
        # 采用较低值，避免其中一个阶段过度自信。
        total_confidence = min(
            coarse_result.confidence,
            fine_result.confidence,
        )

        if fine_result.need_clarification:
            status = IntentStatus.NEED_CLARIFICATION

        elif (
            total_confidence
            < IntentEngine.FINAL_CONFIDENCE_THRESHOLD
        ):
            status = IntentStatus.UNCERTAIN

        else:
            status = IntentStatus.SUCCESS

        return IntentResult(
            status=status,
            coarse=coarse_result,
            fine=fine_result,
            final_intent=fine_result.primary_intent,
            total_confidence=total_confidence,
            need_clarification=fine_result.need_clarification,
            clarification_question=(
                fine_result.clarification_question
            ),
        )


    @staticmethod
    def _coarse_output(coarse_result) -> dict:
        """整理粗粒度识别结果，供 Span 保存。"""

        return {
            "category": coarse_result.category.value,
            "confidence": coarse_result.confidence,
            "need_fine": coarse_result.need_fine,
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in coarse_result.candidates
            ],
            "reason": coarse_result.reason,
        }

    @staticmethod
    def _fine_output(fine_result) -> dict:
        """整理精细识别结果，供 Span 保存。"""

        return fine_result.model_dump(mode="json")

    @staticmethod
    def _final_output(result) -> dict:
        """整理最终意图结果。"""

        return {
            "status": result.status.value,
            "final_intent": result.final_intent,
            "total_confidence": result.total_confidence,
            "need_clarification": result.need_clarification,
            "clarification_question": result.clarification_question,
        }



    async def recognize(self, user_input: str) -> IntentResult:
        """
        意图识别统一入口，并记录完整 Span / Event。

        执行链：
        intent_recognition
            ├── intent_coarse
            └── intent_fine（按需执行）
        """

        user_input = user_input.strip()

        if not user_input:
            raise ValueError("用户输入不能为空")

        coarse_result = None
        fine_result = None

        logger.info(
            "开始意图识别，input_length=%s",
            len(user_input),
        )

        try:
            async with SpanRecorder(
                span_name="intent_recognition",
                span_type="intent",
                input_data={
                    "input_length": len(user_input),
                },
            ) as intent_span:

                # 第一阶段：粗粒度识别
                async with SpanRecorder(
                    span_name="intent_coarse",
                    span_type="llm",
                    input_data={
                        "input_length": len(user_input),
                    },
                ) as coarse_span:

                    coarse_result = (
                        await self.coarse_recognizer.recognize(
                            user_input
                        )
                    )

                    coarse_output = self._coarse_output(
                        coarse_result
                    )

                    coarse_span.set_output(
                        coarse_output
                    )

                await record_event(
                    event_type="intent.coarse.completed",
                    event_data=coarse_output,
                )

                logger.info(
                    "粗粒度识别完成: category=%s, confidence=%.2f",
                    coarse_result.category.value,
                    coarse_result.confidence,
                )

                # 第二阶段：判断是否需要精细识别
                if self._should_run_fine(coarse_result):

                    async with SpanRecorder(
                        span_name="intent_fine",
                        span_type="llm",
                        input_data={
                            "category": coarse_result.category.value,
                            "input_length": len(user_input),
                        },
                    ) as fine_span:

                        fine_result = (
                            await self.fine_recognizer.recognize(
                                user_input=user_input,
                                coarse_result=coarse_result,
                            )
                        )

                        fine_output = self._fine_output(
                            fine_result
                        )

                        fine_span.set_output(
                            fine_output
                        )

                    await record_event(
                        event_type="intent.fine.completed",
                        event_data=fine_output,
                    )

                    final_result = self._build_final_result(
                        coarse_result,
                        fine_result,
                    )

                else:
                    logger.info(
                        "当前请求跳过精细意图识别"
                    )

                    final_result = self._build_simple_result(
                        coarse_result
                    )

                # 记录最终结果
                final_output = self._final_output(
                    final_result
                )

                intent_span.set_output(
                    final_output
                )

                await record_event(
                    event_type="intent.recognition.completed",
                    event_data=final_output,
                )

                return final_result



        except Exception as exc:
            logger.exception("Intent Engine 执行失败")
            await self._safe_record_event(
                event_type="intent.recognition.failed",
                event_data={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                },
            )

            return IntentResult(
                status=IntentStatus.FAILED,
                coarse=coarse_result,
                fine=None,
                final_intent=None,
                total_confidence=0.0,
                need_clarification=False,
                clarification_question=None,
            )


# 全局引擎实例
intent_engine = IntentEngine(
    coarse_recognizer=coarse_intent_recognizer,
    fine_recognizer=fine_intent_recognizer,
)
