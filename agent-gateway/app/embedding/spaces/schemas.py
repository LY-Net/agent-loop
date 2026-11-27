from pydantic import BaseModel, ConfigDict, Field


class EmbeddingSpace(BaseModel):
    """
    业务 Embedding Space 元数据。

    每个 Space 绑定一个确定的模型配置指纹，
    避免不同模型配置共用同一个向量空间。
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    space_id: str = Field(min_length=1)

    business_name: str = Field(min_length=1)

    description: str = ""

    provider_name: str = Field(min_length=1)

    model_name: str = Field(min_length=1)

    model_version: str = Field(min_length=1)

    dimension: int = Field(gt=0)

    # SHA256 指纹，共 64 位十六进制字符
    fingerprint: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    index_name: str = Field(min_length=1)
