import math
import re

from app.embedding.schemas import EmbeddingInput
from app.embedding.spaces.service import EmbeddingSpaceService

from app.tools.catalog.schemas import ToolDiscoveryQuery
from app.tools.discovery.repository import ToolDiscoveryRepository
from app.tools.discovery.schemas import (
    ToolDiscoveryResult,
    ToolDiscoveryResponse,
)


TOOL_DISCOVERY_SPACE_ID = "tool_discovery"
RRF_K = 60


class ToolDiscoveryService:
    """
    工具发现引擎。

    当前检索流程：
    1. Query Understanding（规则化处理）
    2. Vector Retrieval
    3. Keyword Retrieval
    4. RRF Fusion
    5. Top-K 返回
    """

    def __init__(
        self,
        repository: ToolDiscoveryRepository,
        embedding_space_service: EmbeddingSpaceService,
    ):
        self.repository = repository
        self.embedding_space_service = embedding_space_service

    async def discover(
        self,
        query: ToolDiscoveryQuery,
    ) -> ToolDiscoveryResponse:

        if query.top_k < 1 or query.top_k > 100:
            raise ValueError("top_k 必须在 1 到 100 之间")

        normalized_query = query.query.strip()

        # 当前版本不对空查询执行全量向量召回
        if not normalized_query and not query.keywords:
            return ToolDiscoveryResponse(
                query=query.query,
                total=0,
                results=[],
            )

        # 1. Query Understanding
        terms = self._extract_terms(
            normalized_query,
            query.keywords,
        )

        candidate_k = min(
            max(query.top_k * 4, 20),
            200,
        )

        vector_results = []
        keyword_results = []

        # 2. Vector Retrieval
        if normalized_query:
            vectors = await self.embedding_space_service.embed(
                space_id=TOOL_DISCOVERY_SPACE_ID,
                request=EmbeddingInput(
                    texts=[normalized_query],
                ),
            )

            if len(vectors) != 1:
                raise RuntimeError(
                    "Query Embedding 返回数量错误"
                )

            query_vector = vectors[0]
            space = self.embedding_space_service.get_space(
                TOOL_DISCOVERY_SPACE_ID
            )

            if len(query_vector) != space.dimension:
                raise RuntimeError(
                    "Query Embedding 维度错误"
                )

            if not all(
                math.isfinite(value)
                for value in query_vector
            ):
                raise RuntimeError(
                    "Query Embedding 包含非有限值"
                )

            vector_results = (
                await self.repository.search_vector(
                    query_vector=query_vector,
                    space_id=TOOL_DISCOVERY_SPACE_ID,
                    model_fingerprint=(
                        self.embedding_space_service
                        .model_info.fingerprint
                    ),
                    provider_types=query.provider_types,
                    capabilities=query.capabilities,
                    limit=candidate_k,
                )
            )

        # 3. Keyword Retrieval
        if terms:
            keyword_results = (
                await self.repository.search_keyword(
                    terms=terms,
                    provider_types=query.provider_types,
                    capabilities=query.capabilities,
                    limit=candidate_k,
                )
            )

        # 4. RRF Fusion
        fused = {}

        for rank, item in enumerate(vector_results, start=1):
            tool_id = item["tool_id"]

            entry = fused.setdefault(
                tool_id,
                {
                    "data": item,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "keyword_rank": None,
                    "distance": None,
                    "matched_by": [],
                },
            )

            entry["rrf_score"] += 1.0 / (RRF_K + rank)
            entry["vector_rank"] = rank
            entry["distance"] = float(item["distance"])

            if "vector" not in entry["matched_by"]:
                entry["matched_by"].append("vector")

        for rank, item in enumerate(keyword_results, start=1):
            tool_id = item["tool_id"]

            entry = fused.setdefault(
                tool_id,
                {
                    "data": item,
                    "rrf_score": 0.0,
                    "vector_rank": None,
                    "keyword_rank": None,
                    "distance": None,
                    "matched_by": [],
                },
            )

            entry["rrf_score"] += 1.0 / (RRF_K + rank)
            entry["keyword_rank"] = rank

            if "keyword" not in entry["matched_by"]:
                entry["matched_by"].append("keyword")

        # 5. 排序并截取 Top-K
        ranked_items = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )[:query.top_k]

        results = []

        for item in ranked_items:
            data = item["data"]
            distance = item["distance"]

            results.append(
                ToolDiscoveryResult(
                    tool_id=data["tool_id"],
                    server_id=data["server_id"],
                    provider_type=data["provider_type"],
                    name=data["name"],
                    description=data["description"] or "",
                    capabilities=data["capabilities"] or [],
                    keywords=data["keywords"] or [],
                    status=data["status"],
                    distance=distance,
                    similarity=(
                        1.0 - distance
                        if distance is not None
                        else None
                    ),
                    vector_rank=item["vector_rank"],
                    keyword_rank=item["keyword_rank"],
                    rrf_score=item["rrf_score"],
                    matched_by=item["matched_by"],
                )
            )

        return ToolDiscoveryResponse(
            query=query.query,
            total=len(results),
            results=results,
        )

    @staticmethod
    def _extract_terms(
        query: str,
        keywords: list[str],
    ) -> list[str]:
        """
        第一版规则化关键词提取。

        - 英文按单词提取。
        - 中文保留连续文本，并生成相邻双字词。
        - 合并调用方显式传入的 keywords。
        """

        terms = []

        if query:
            terms.append(query)

            tokens = re.findall(
                r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+",
                query.lower(),
            )

            for token in tokens:
                if token not in terms:
                    terms.append(token)

                # 中文连续文本生成双字片段，
                # 用于提高描述中的局部匹配机会。
                if re.fullmatch(
                    r"[\u4e00-\u9fff]+",
                    token,
                ) and len(token) > 2:
                    for index in range(len(token) - 1):
                        gram = token[index:index + 2]

                        if gram not in terms:
                            terms.append(gram)

        for keyword in keywords:
            keyword = keyword.strip().lower()

            if keyword and keyword not in terms:
                terms.append(keyword)

        return terms