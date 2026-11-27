from pydantic import BaseModel, ConfigDict, Field

from app.tools.catalog.schemas import (
    ToolProviderType,
    CatalogToolStatus,
)


class ToolDiscoveryResult(BaseModel):
    """工具发现结果，不包含完整 input_schema。"""

    model_config = ConfigDict(from_attributes=True)

    tool_id: str
    server_id: str | None = None
    provider_type: ToolProviderType
    name: str
    description: str = ""
    capabilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    status: CatalogToolStatus

    distance: float | None = None
    similarity: float | None = None

    vector_rank: int | None = None
    keyword_rank: int | None = None
    rrf_score: float

    matched_by: list[str] = Field(default_factory=list)


class ToolDiscoveryResponse(BaseModel):
    query: str
    total: int
    results: list[ToolDiscoveryResult] = Field(
        default_factory=list
    )