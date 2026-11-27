from pydantic import BaseModel, ConfigDict, Field


class RagDocumentIngestRequest(BaseModel):
    """RAG 文档入库请求。"""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=512)
    content: str = Field(min_length=1)
    source: str = Field(default="", max_length=1024)
    metadata: dict = Field(default_factory=dict)

    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0, le=1000)


class RagDocumentIngestResponse(BaseModel):
    """RAG 文档入库响应。"""

    document_id: str
    chunk_count: int
    embedding_dimension: int
    model_fingerprint: str


class RagDocumentDeleteResponse(BaseModel):
    """RAG 文档删除响应。"""

    document_id: str
    deleted: bool


class RagSearchRequest(BaseModel):
    """RAG 向量检索请求。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=100)


class RagSearchResult(BaseModel):
    """RAG 混合检索结果。"""

    document_id: str
    title: str
    source: str
    chunk_index: int
    content: str

    # 向量检索指标；仅关键词命中的结果可能为空
    distance: float | None = None
    similarity: float | None = None

    # 混合检索指标
    rrf_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    matched_by: list[str] = Field(default_factory=list)


class RagSearchResponse(BaseModel):
    """RAG 向量检索响应。"""

    query: str
    results: list[RagSearchResult]