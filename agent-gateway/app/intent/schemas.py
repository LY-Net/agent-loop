
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


# ==========================================
# 1. 意图分类枚举
# ==========================================

class IntentCategory(str, Enum):
    """
    一级意图分类
    """

    GENERAL = "general"                 # 普通对话
    KNOWLEDGE_QA = "knowledge_qa"       # 知识问答
    DATA_ANALYSIS = "data_analysis"     # 数据分析
    DATA_QUERY = "data_query"           # 数据查询
    CONTENT_CREATION = "content_creation"  # 内容生成
    TASK_EXECUTION = "task_execution"   # 任务执行
    SYSTEM_OPERATION = "system_operation"  # 系统操作
    OTHER = "other"                     # 其他


class IntentStatus(str, Enum):
    """
    意图识别状态
    """

    SUCCESS = "success"
    UNCERTAIN = "uncertain"
    NEED_CLARIFICATION = "need_clarification"
    FAILED = "failed"


class ExecutionMode(str, Enum):
    """
    Agent 执行模式
    """

    DIRECT = "direct"       # 直接回答
    RAG = "rag"             # 知识检索
    TOOL = "tool"           # 工具调用
    HYBRID = "hybrid"       # RAG + Tool
    PLANNING = "planning"   # 复杂任务规划


# ==========================================
# 2. 初筛数据模型
# ==========================================

class IntentCandidate(BaseModel):
    """
    初筛阶段识别出的候选意图
    """

    model_config = ConfigDict(extra="forbid")

    intent_name: str = Field(
        description="候选意图名称"
    )

    description: str = Field(
        default="",
        description="候选意图描述"
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="候选意图置信度"
    )


class CoarseIntentResult(BaseModel):
    """
    意图初筛结果
    """

    model_config = ConfigDict(extra="forbid")

    category: IntentCategory = Field(
        description="用户请求所属一级分类"
    )

    candidates: list[IntentCandidate] = Field(
        default_factory=list,
        description="候选意图列表"
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="初筛整体置信度"
    )

    need_fine: bool = Field(
        default=True,
        description="是否需要进入精筛"
    )

    reason: str = Field(
        default="",
        description="初筛判断依据"
    )


# ==========================================
# 3. 精筛数据模型
# ==========================================

class IntentEntity(BaseModel):
    """
    从用户输入中提取的实体
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        description="实体名称"
    )

    value: Any = Field(
        description="实体值"
    )

    entity_type: str = Field(
        default="unknown",
        description="实体类型"
    )


class IntentFineResult(BaseModel):
    """
    意图精筛结果
    """

    model_config = ConfigDict(extra="forbid")

    primary_intent: str = Field(
        description="最终识别出的具体意图"
    )

    intent_description: str = Field(
        default="",
        description="具体意图描述"
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="精筛置信度"
    )

    entities: list[IntentEntity] = Field(
        default_factory=list,
        description="识别出的业务实体"
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="任务执行参数"
    )

    required_capabilities: list[str] = Field(
        default_factory=list,
        description="执行任务所需能力"
    )

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.DIRECT,
        description="建议采用的执行模式"
    )

    need_rag: bool = Field(
        default=False,
        description="是否需要知识检索"
    )

    need_tool: bool = Field(
        default=False,
        description="是否需要调用工具"
    )

    need_clarification: bool = Field(
        default=False,
        description="是否需要向用户澄清"
    )

    clarification_question: str | None = Field(
        default=None,
        description="需要澄清时向用户提出的问题"
    )

    reason: str = Field(
        default="",
        description="精筛判断依据"
    )


# ==========================================
# 4. 完整意图识别结果
# ==========================================

class IntentResult(BaseModel):
    """
    完整意图识别结果

    包含初筛和精筛两个阶段的数据。
    """

    model_config = ConfigDict(extra="forbid")

    status: IntentStatus = Field(
        default=IntentStatus.SUCCESS,
        description="意图识别状态"
    )

    coarse: CoarseIntentResult = Field(
        description="意图初筛结果"
    )

    fine: IntentFineResult | None = Field(
        default=None,
        description="意图精筛结果"
    )

    final_intent: str | None = Field(
        default=None,
        description="最终意图"
    )

    total_confidence: float = Field(
        default=0,
        ge=0,
        le=1,
        description="整体意图识别置信度"
    )

    need_clarification: bool = Field(
        default=False,
        description="是否需要澄清"
    )

    clarification_question: str | None = Field(
        default=None,
        description="澄清问题"
    )
