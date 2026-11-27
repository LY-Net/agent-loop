
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.embedding.policy import DataClassification


class EmbeddingInput(BaseModel):
    """
    Embedding 统一输入模型。

    所有业务模块通过该模型向 EmbeddingService 提交请求。
    Provider 不直接接收该对象。
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    texts: list[str] = Field(
        min_length=1,
        description="需要生成 Embedding 的文本列表",
    )

    classification: DataClassification = Field(
        default=DataClassification.INTERNAL,
        description="输入数据的安全分类",
    )

    allow_external: bool = Field(
        default=False,
        description="是否明确允许将数据发送到外部服务",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(
                    f"texts[{index}] cannot be empty or whitespace only"
                )

        return texts



class EmbeddingModelInfo(BaseModel):
    """
    Embedding 模型元数据。

    fingerprint 用于标识模型配置，
    避免不同模型配置被误认为兼容。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    provider_name: str = Field(
        min_length=1,
        description="Provider 标识",
    )

    model_name: str = Field(
        min_length=1,
        description="Embedding 模型名称",
    )

    model_version: str = Field(
        min_length=1,
        description="模型版本或权重标识",
    )

    dimension: int = Field(
        gt=0,
        description="Embedding 向量维度",
    )

    is_remote: bool = Field(
        description="是否为远程 Provider",
    )

    normalize_embeddings: bool = Field(
        default=True,
        description="是否对 Embedding 向量进行归一化",
    )

    fingerprint: str = Field(
        min_length=1,
        description="模型配置指纹",
    )
