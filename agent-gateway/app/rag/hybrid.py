
import re

from sqlalchemy import select, func, literal
from sqlalchemy.ext.asyncio import AsyncSession

from app.embedding.spaces.registry import EmbeddingSpaceRegistry
from app.embedding.spaces.service import EmbeddingSpaceService

from app.models.rag import RagChunk, RagDocument

from app.rag.retrieval import RagRetrievalService
from app.rag.schemas import (
    RagSearchRequest,
    RagSearchResponse,
    RagSearchResult,
)


RAG_SPACE_ID = "rag_documents"
RRF_K = 60


def build_fts_query(query: str) -> str:
    """
    提取英文、数字和技术名称，构造 PostgreSQL OR 查询。

    例如：
        PostgreSQL HNSW 向量检索
    转换为：
        PostgreSQL | HNSW

    中文连续文本由 Trigram 检索负责。
    """

    tokens = re.findall(
        r"[A-Za-z0-9_]+",
        query,
    )

    tokens = list(dict.fromkeys(tokens))

    if not tokens:
        return ""

    return " | ".join(tokens)


class RagHybridSearchService:
    """向量 + 关键词混合检索服务。"""

    def __init__(
        self,
        session: AsyncSession,
        embedding_space_service: EmbeddingSpaceService,
        space_registry: EmbeddingSpaceRegistry,
    ) -> None:
        self.session = session
        self.embedding_space_service = embedding_space_service
        self.space_registry = space_registry

        self.vector_search = RagRetrievalService(
            session=session,
            embedding_space_service=embedding_space_service,
            space_registry=space_registry,
        )

    async def search(
        self,
        request: RagSearchRequest,
    ) -> RagSearchResponse:

        candidate_k = min(
            max(request.top_k * 4, 20),
            200,
        )

        fingerprint = self._current_fingerprint()

        # ==========================================
        # 1. 向量检索
        # ==========================================

        vector_response = await self.vector_search.search(
            RagSearchRequest(
                query=request.query,
                top_k=candidate_k,
            )
        )

        vector_results = vector_response.results

        vector_ranks = {
            self._key(item): rank
            for rank, item in enumerate(
                vector_results,
                start=1,
            )
        }

        vector_items = {
            self._key(item): item
            for item in vector_results
        }

        # ==========================================
        # 2. PostgreSQL 全文检索 FTS
        # ==========================================

        fts_rows = []

        fts_query = build_fts_query(request.query)

        if fts_query:

            tsvector = func.to_tsvector(
                "simple",
                RagChunk.content,
            )

            tsquery = func.to_tsquery(
                "simple",
                fts_query,
            )

            fts_rank = func.ts_rank_cd(
                tsvector,
                tsquery,
            )

            fts_stmt = (
                select(
                    RagChunk,
                    RagDocument,
                    fts_rank.label("score"),
                )
                .join(
                    RagDocument,
                    RagChunk.document_id == RagDocument.id,
                )
                .where(
                    tsvector.op("@@")(tsquery),
                    RagChunk.space_id == RAG_SPACE_ID,
                    RagChunk.model_fingerprint == fingerprint,
                    RagDocument.status == "active",
                )
                .order_by(
                    fts_rank.desc(),
                    RagChunk.id,
                )
                .limit(candidate_k)
            )

            fts_result = await self.session.execute(
                fts_stmt
            )

            fts_rows = fts_result.all()

        # ==========================================
        # 3. Trigram 中文关键词检索
        # ==========================================

        query_expression = literal(request.query)

        trigram_score = func.word_similarity(
            query_expression,
            RagChunk.content,
        )

        # query <% content
        # 判断查询是否与较长文本中的某个连续部分相似
        trigram_condition = query_expression.op("<%")(
            RagChunk.content
        )

        trigram_stmt = (
            select(
                RagChunk,
                RagDocument,
                trigram_score.label("score"),
            )
            .join(
                RagDocument,
                RagChunk.document_id == RagDocument.id,
            )
            .where(
                trigram_condition,
                RagChunk.space_id == RAG_SPACE_ID,
                RagChunk.model_fingerprint == fingerprint,
                RagDocument.status == "active",
            )
            .order_by(
                trigram_score.desc(),
                RagChunk.id,
            )
            .limit(candidate_k)
        )

        trigram_result = await self.session.execute(
            trigram_stmt
        )

        trigram_rows = trigram_result.all()

        # ==========================================
        # 4. 合并两路关键词排名
        # ==========================================

        lexical_scores: dict[tuple, float] = {}
        lexical_sources: dict[tuple, set[str]] = {}
        lexical_records: dict[tuple, tuple] = {}

        for source_name, rows in (
            ("fts", fts_rows),
            ("trigram", trigram_rows),
        ):
            for rank, (chunk, document, score) in enumerate(
                rows,
                start=1,
            ):
                key = (
                    document.document_id,
                    chunk.chunk_index,
                )

                lexical_scores[key] = (
                    lexical_scores.get(key, 0.0)
                    + 1.0 / (RRF_K + rank)
                )

                lexical_sources.setdefault(
                    key,
                    set(),
                ).add(source_name)

                lexical_records[key] = (
                    chunk,
                    document,
                )

        lexical_order = sorted(
            lexical_scores,
            key=lambda key: (
                -lexical_scores[key],
                key,
            ),
        )

        lexical_ranks = {
            key: rank
            for rank, key in enumerate(
                lexical_order,
                start=1,
            )
        }

        # ==========================================
        # 5. RRF 融合向量和关键词排名
        # ==========================================

        all_keys = (
            set(vector_ranks)
            | set(lexical_ranks)
        )

        merged: list[
            tuple[float, tuple, RagSearchResult]
        ] = []

        for key in all_keys:

            vector_rank = vector_ranks.get(key)
            keyword_rank = lexical_ranks.get(key)

            score = 0.0

            if vector_rank is not None:
                score += 1.0 / (
                    RRF_K + vector_rank
                )

            if keyword_rank is not None:
                score += 1.0 / (
                    RRF_K + keyword_rank
                )

            vector_item = vector_items.get(key)

            if vector_item is not None:

                result = vector_item.model_copy(
                    update={
                        "vector_rank": vector_rank,
                        "keyword_rank": keyword_rank,
                        "rrf_score": score,
                        "matched_by": (
                            ["vector"]
                            + sorted(
                                lexical_sources.get(
                                    key,
                                    set(),
                                )
                            )
                        ),
                    }
                )

            else:

                chunk, document = lexical_records[key]

                result = RagSearchResult(
                    document_id=document.document_id,
                    title=document.title,
                    source=document.source,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    rrf_score=score,
                    keyword_rank=keyword_rank,
                    matched_by=sorted(
                        lexical_sources.get(
                            key,
                            set(),
                        )
                    ),
                )

            merged.append(
                (score, key, result)
            )

        merged.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        final_results = [
            item[2]
            for item in merged[:request.top_k]
        ]

        return RagSearchResponse(
            query=request.query,
            results=final_results,
        )

    def _current_fingerprint(self) -> str:

        space = self.space_registry.get(
            RAG_SPACE_ID
        )

        if space is None:
            raise RuntimeError(
                f"Embedding Space not found: {RAG_SPACE_ID}"
            )

        return space.fingerprint

    @staticmethod
    def _key(
        item: RagSearchResult,
    ) -> tuple:

        return (
            item.document_id,
            item.chunk_index,
        )
