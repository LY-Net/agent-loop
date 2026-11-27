
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field
from uuid import uuid4


class PlanStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEED_CLARIFICATION = "need_clarification"


class StepType(str, Enum):
    CLARIFY = "clarify"
    RAG_RETRIEVAL = "rag_retrieval"
    TOOL_CALL = "tool_call"
    LLM_GENERATION = "llm_generation"
    TASK_PLANNING = "task_planning"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionStep(BaseModel):
    step_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    step_type: StepType

    name: str

    description: str = ""

    # 当前步骤依赖哪些步骤
    depends_on: list[str] = Field(default_factory=list)

    # 当前步骤的执行参数
    input_data: dict[str, Any] = Field(default_factory=dict)

    # 执行结果
    output_data: dict[str, Any] = Field(default_factory=dict)

    status: StepStatus = StepStatus.PENDING

    required_capabilities: list[str] = Field(
        default_factory=list
    )


class ExecutionPlan(BaseModel):
    plan_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    intent_name: str

    execution_mode: str

    description: str = ""

    status: PlanStatus = PlanStatus.CREATED

    steps: list[ExecutionStep] = Field(
        default_factory=list
    )

    # 是否需要用户进一步确认
    need_clarification: bool = False

    clarification_question: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ExecutionMode(BaseModel):
    TOOL:str