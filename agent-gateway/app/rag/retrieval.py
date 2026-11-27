import math

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding.schemas import EmbeddingInput
from app.embedding.spaces.registry import EmbeddingSpaceRegistry
from app.embedding.spaces.service import EmbeddingSpaceService

from app.models.rag import RagChunk, RagDocument

from app.rag.schemas import (
    RagSearchRequest,
    RagSearchResponse,
    RagSearchResult,
)


RAG_SPACE_ID = "rag_documents"


class RagRetrievalService:
    """RAG 向量检索服务。"""

    def __init__(
        self,
        session: AsyncSession,
        embedding_space_service: EmbeddingSpaceService,
        space_registry: EmbeddingSpaceRegistry,
    ) -> None:
        self.session = session
        self.embedding_space_service = embedding_space_service
        self.space_registry = space_registry

    async def search(
        self,
        request: RagSearchRequest,
    ) -> RagSearchResponse:
        # 1. 获取 Space 元数据
        space = self.space_registry.get(RAG_SPACE_ID)

        if space is None:
            raise RuntimeError(
                f"Embedding Space not found: {RAG_SPACE_ID}"
            )

        # 2. 将用户问题转换成向量
        vectors = await self.embedding_space_service.embed(
            space_id=RAG_SPACE_ID,
            request=EmbeddingInput(texts=[request.query]),
        )

        if len(vectors) != 1:
            raise RuntimeError("Query embedding count mismatch")

        query_vector = vectors[0]

        if len(query_vector) != space.dimension:
            raise RuntimeError("Query embedding dimension mismatch")

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in query_vector
        ):
            raise RuntimeError("Invalid query embedding vector")

        # 3. 构造 Cosine Distance 查询
        distance = RagChunk.embedding.cosine_distance(
            query_vector
        )

        stmt = (
            select(
                RagChunk,
                RagDocument,
                distance.label("distance"),
            )
            .join(
                RagDocument,
                RagChunk.document_id == RagDocument.id,
            )
            .where(
                RagChunk.space_id == RAG_SPACE_ID,
                RagChunk.model_fingerprint == space.fingerprint,
                RagDocument.status == "active",
            )
            # 保留原始 distance 排序，便于 PostgreSQL 使用 HNSW。
            .order_by(distance)
            .limit(request.top_k)
        )

        result = await self.session.execute(stmt)

        rows = result.all()

        # 4. 组装检索结果
        search_results: list[RagSearchResult] = []

        for chunk, document, raw_distance in rows:
            distance_value = float(raw_distance)

            search_results.append(
                RagSearchResult(
                    document_id=document.document_id,
                    title=document.title,
                    source=document.source,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    distance=distance_value,
                    similarity=1.0 - distance_value,
                )
            )

        return RagSearchResponse(
            query=request.query,
            results=search_results,
        )