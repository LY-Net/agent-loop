
import asyncio
import re
import math
from uuid import uuid4

from sqlalchemy import select, func, literal

from app.main import app
from app.db.session import AsyncSessionLocal
from app.models.rag import RagChunk, RagDocument

from app.rag.service import RagService
from app.rag.retrieval import RagRetrievalService
from app.rag.hybrid import RagHybridSearchService
from app.rag.schemas import (
    RagDocumentIngestRequest,
    RagSearchRequest,
)


TOP_K = 5
RRF_K = 60
CANDIDATE_K = 100


# =========================================================
# 1. 评估文档：20 篇
# =========================================================

DOCUMENTS = [
    # PostgreSQL / pgvector：5 篇
    {
        "key": "postgresql_hnsw",
        "title": "PostgreSQL HNSW 向量索引",
        "content": (
            "PostgreSQL 使用 pgvector 扩展支持向量存储和相似度搜索。"
            "HNSW 是一种基于分层图结构的近似最近邻搜索索引。"
            "它适合高维向量检索，可以配置 m、ef_construction 和 ef_search。"
            "使用 cosine distance 可以计算两个向量之间的余弦距离。"
        ),
    },
    {
        "key": "postgresql_btree",
        "title": "PostgreSQL B-tree 索引",
        "content": (
            "PostgreSQL 默认使用 B-tree 索引处理大量结构化数据查询。"
            "B-tree 适合等值查询、范围查询、排序和比较操作。"
            "例如用户 ID、创建时间和状态字段都可以建立 B-tree 索引。"
            "B-tree 与 HNSW 不同，它主要服务于关系型字段查询。"
        ),
    },
    {
        "key": "postgresql_gin",
        "title": "PostgreSQL GIN 全文检索索引",
        "content": (
            "PostgreSQL GIN 索引可以加速全文搜索和 JSONB 查询。"
            "全文检索可以通过 to_tsvector 和 plainto_tsquery 实现。"
            "GIN 索引能够保存文档中的词项信息，帮助数据库快速定位匹配内容。"
            "它与向量 HNSW 索引的用途不同，但都能提升数据检索效率。"
        ),
    },
    {
        "key": "pgvector_ivfflat",
        "title": "pgvector IVFFlat 向量索引",
        "content": (
            "pgvector 支持 IVFFlat 近似最近邻向量索引。"
            "IVFFlat 通过聚类将向量划分到不同的倒排列表。"
            "查询时可以调整 probes 参数控制搜索的列表数量。"
            "IVFFlat 和 HNSW 都能进行向量相似度搜索，"
            "但索引结构和构建方式不同。"
        ),
    },
    {
        "key": "pgvector_overview",
        "title": "pgvector 向量数据库扩展",
        "content": (
            "pgvector 是 PostgreSQL 的向量扩展，支持保存和检索 Embedding。"
            "它支持 L2、内积和余弦距离等向量运算。"
            "pgvector 提供 HNSW 和 IVFFlat 等近似最近邻索引，"
            "可以将向量搜索能力集成到现有 PostgreSQL 数据库。"
        ),
    },

    # Embedding / BGE-M3：3 篇
    {
        "key": "bge_model",
        "title": "BGE-M3 多语言 Embedding 模型",
        "content": (
            "BGE-M3 是一个多语言文本 Embedding 模型。"
            "Agent Gateway 当前使用 BGE-M3 Dense Embedding，"
            "输出向量维度为 1024。"
            "模型能够将中文和英文文本转换成稠密向量，"
            "用于语义搜索和 RAG 知识检索。"
        ),
    },
    {
        "key": "embedding_service",
        "title": "Embedding Service 向量生成服务",
        "content": (
            "EmbeddingService 负责接收文本并调用对应的 Embedding Provider。"
            "服务需要校验输入、批量处理文本、检查向量数量和维度。"
            "Embedding Service 还负责执行数据分类策略和 Provider 路由，"
            "而不是让底层 Provider 自己决定数据是否可以外发。"
        ),
    },
    {
        "key": "bge_runtime",
        "title": "BGE-M3 本地模型运行配置",
        "content": (
            "BGE-M3 可以部署在本地 CPU 环境中运行。"
            "当前模型使用 FP32 精度，输出 1024 维 Dense Vector。"
            "FlagEmbedding 负责加载模型并生成向量，"
            "Embedding Server 通过 HTTP 接口提供模型能力。"
            "模型权重、设备类型、归一化和运行参数都属于模型运行配置。"
        ),
    },

    # Agent / MCP：3 篇
    {
        "key": "gateway_discovery",
        "title": "Agent Gateway MCP 工具发现",
        "content": (
            "Agent Gateway 可以连接 MCP Server 并自动发现可用工具。"
            "连接成功后，Gateway 获取工具名称、描述和输入 Schema，"
            "将工具元数据同步到 Tool Catalog。"
            "Tool Discovery 负责根据用户请求检索合适的工具。"
        ),
    },
    {
        "key": "gateway_dispatch",
        "title": "Agent Gateway 工具调度与执行",
        "content": (
            "Agent Gateway 根据用户请求选择适合的工具，"
            "然后将参数交给对应的 Tool Handler 执行。"
            "工具调度需要处理参数校验、执行结果、异常和调用追踪。"
            "模型调用负责理解意图，工具调度负责实际执行。"
        ),
    },
    {
        "key": "mcp_lifecycle",
        "title": "MCP Server 生命周期管理",
        "content": (
            "MCP Server 接入后需要完成连接、初始化、工具发现和同步。"
            "当 MCP Server 的工具发生变化时，Gateway 应更新 Tool Catalog。"
            "服务断开时需要处理连接状态和资源释放。"
            "MCP 协议定义工具交互方式，但不直接负责 Gateway 的业务调度策略。"
        ),
    },

    # 数据分类策略：3 篇
    {
        "key": "policy_outbound",
        "title": "Embedding 数据外发策略",
        "content": (
            "Embedding 数据外发策略决定文本是否允许发送到云端模型服务。"
            "敏感数据和内部数据默认使用本地 Embedding Provider。"
            "只有明确允许外发的数据，才可以调用云端 Embedding API。"
            "数据外发控制需要发生在调用模型 Provider 之前。"
        ),
    },
    {
        "key": "policy_classification",
        "title": "文本数据分类与安全等级",
        "content": (
            "文本数据可以根据业务安全要求划分为公开、内部和敏感等级。"
            "不同分类的数据具有不同的处理权限。"
            "敏感信息需要限制访问和传输，不能默认交给外部模型处理。"
            "数据分类结果应成为后续 Embedding 路由的输入条件。"
        ),
    },
    {
        "key": "provider_routing",
        "title": "Embedding Provider 路由机制",
        "content": (
            "Embedding Provider Registry 负责管理可用的向量模型 Provider。"
            "EmbeddingService 根据数据分类和外发权限选择本地或远程 Provider。"
            "本地 Provider 可以调用内网部署的 BGE-M3，"
            "远程 Provider 可以调用经过授权的云端 API。"
            "Provider 本身只负责向量生成，不承担数据分类决策。"
        ),
    },

    # Embedding Space：3 篇
    {
        "key": "space_isolation",
        "title": "Embedding Space 业务数据隔离",
        "content": (
            "Embedding Space 用于隔离不同业务的向量数据。"
            "RAG Documents 和 Tool Discovery 可以拥有独立的逻辑 Space。"
            "不同 Space 保存各自的业务数据和索引配置，"
            "同时允许多个 Space 共用同一个 BGE-M3 模型。"
        ),
    },
    {
        "key": "space_registry",
        "title": "Embedding Space Registry 注册管理",
        "content": (
            "Embedding Space Registry 负责注册和查询业务向量空间。"
            "注册时需要校验 Provider、模型名称、版本、维度和指纹。"
            "Space 元数据可以持久化在 PostgreSQL 中，"
            "应用启动时恢复到内存 Registry。"
        ),
    },
    {
        "key": "provider_shared",
        "title": "多个业务共享 Embedding Provider",
        "content": (
            "多个业务可以共用同一个 Embedding Provider 和模型实例。"
            "RAG 与 Tool Discovery 使用不同的逻辑 Space，"
            "但可以共用 local_bge Provider 和 BGE-M3 模型。"
            "共享模型能够减少重复部署，独立 Space 能够保持业务数据隔离。"
        ),
    },

    # RAG Pipeline：3 篇
    {
        "key": "rag_ingestion",
        "title": "RAG 文档切片与向量入库",
        "content": (
            "RAG 文档入库首先对原始文本进行切片。"
            "随后调用 Embedding Service 生成文本向量。"
            "每个 Chunk 保存文本内容、向量、所属文档和 Embedding Space。"
            "向量最终持久化到 PostgreSQL 的 pgvector 字段中。"
        ),
    },
    {
        "key": "rag_chunking",
        "title": "RAG 文本切片与 Chunk 重叠策略",
        "content": (
            "RAG 文档切片需要控制 Chunk Size 和 Chunk Overlap。"
            "过大的 Chunk 会增加上下文噪声，过小的 Chunk 可能丢失语义。"
            "相邻 Chunk 之间保留一定重叠，可以减少切片边界造成的信息损失。"
            "切片时可以优先考虑段落、句子和自然文本边界。"
        ),
    },
    {
        "key": "rag_retrieval",
        "title": "RAG 混合检索与知识召回",
        "content": (
            "RAG 检索可以结合 Dense Vector Search 和 Keyword Search。"
            "向量检索负责语义相似度召回，全文检索负责关键词匹配。"
            "多个检索通道可以通过 Reciprocal Rank Fusion 进行排名融合。"
            "最终根据融合分数返回 Top-K 知识 Chunk。"
        ),
    },
]


# =========================================================
# 2. 查询集：12 条
# =========================================================

EVALUATION_CASES = [
    {
        "query": "PostgreSQL 如何使用 HNSW 加速向量相似度搜索",
        "category": "PostgreSQL",
        "relevant": {
            "postgresql_hnsw": 3,
            "pgvector_overview": 2,
        },
        "hard_negatives": [
            "postgresql_btree",
            "postgresql_gin",
            "pgvector_ivfflat",
        ],
    },
    {
        "query": "PostgreSQL 常见索引类型 B-tree 和 GIN 有什么区别",
        "category": "PostgreSQL",
        "relevant": {
            "postgresql_btree": 3,
            "postgresql_gin": 3,
        },
        "hard_negatives": [
            "postgresql_hnsw",
            "pgvector_ivfflat",
        ],
    },
    {
        "query": "pgvector 的 IVFFlat 和 HNSW 向量索引有什么区别",
        "category": "PostgreSQL",
        "relevant": {
            "pgvector_ivfflat": 3,
            "postgresql_hnsw": 2,
            "pgvector_overview": 2,
        },
        "hard_negatives": [
            "postgresql_btree",
            "postgresql_gin",
        ],
    },
    {
        "query": "BGE-M3 能否处理中文和英文，生成多少维向量",
        "category": "Embedding",
        "relevant": {
            "bge_model": 3,
            "bge_runtime": 2,
        },
        "hard_negatives": [
            "embedding_service",
            "provider_routing",
        ],
    },
    {
        "query": "如何通过本地 CPU 服务批量生成文本 Embedding",
        "category": "Embedding",
        "relevant": {
            "bge_runtime": 3,
            "embedding_service": 3,
            "bge_model": 2,
        },
        "hard_negatives": [
            "provider_routing",
            "space_registry",
        ],
    },
    {
        "query": "如何自动发现 MCP Server 提供的工具并同步 Schema",
        "category": "Agent",
        "relevant": {
            "gateway_discovery": 3,
            "mcp_lifecycle": 2,
        },
        "hard_negatives": [
            "gateway_dispatch",
            "rag_retrieval",
        ],
    },
    {
        "query": "Agent 如何根据用户意图选择工具并执行 Handler",
        "category": "Agent",
        "relevant": {
            "gateway_dispatch": 3,
            "gateway_discovery": 2,
        },
        "hard_negatives": [
            "mcp_lifecycle",
            "provider_routing",
        ],
    },
    {
        "query": "哪些敏感文本不能发送到外部云端模型",
        "category": "Policy",
        "relevant": {
            "policy_outbound": 3,
            "policy_classification": 3,
            "provider_routing": 2,
        },
        "hard_negatives": [
            "bge_runtime",
            "embedding_service",
        ],
    },
    {
        "query": "如何根据数据安全等级选择本地或远程 Embedding Provider",
        "category": "Policy",
        "relevant": {
            "provider_routing": 3,
            "policy_classification": 3,
            "policy_outbound": 2,
        },
        "hard_negatives": [
            "bge_runtime",
            "space_registry",
        ],
    },
    {
        "query": "RAG 和 Tool Discovery 如何共享模型同时隔离向量数据",
        "category": "Embedding Space",
        "relevant": {
            "space_isolation": 3,
            "provider_shared": 3,
            "space_registry": 2,
        },
        "hard_negatives": [
            "embedding_service",
            "provider_routing",
        ],
    },
    {
        "query": "Embedding Space 如何注册并校验模型维度和指纹",
        "category": "Embedding Space",
        "relevant": {
            "space_registry": 3,
            "space_isolation": 2,
            "provider_shared": 1,
        },
        "hard_negatives": [
            "bge_runtime",
            "embedding_service",
        ],
    },
    {
        "query": "如何切分知识文档并结合向量检索和关键词检索",
        "category": "RAG",
        "relevant": {
            "rag_ingestion": 3,
            "rag_chunking": 3,
            "rag_retrieval": 3,
        },
        "hard_negatives": [
            "postgresql_hnsw",
            "pgvector_overview",
        ],
    },
]


# =========================================================
# 3. 关键词提取
# =========================================================

def build_fts_query(query: str) -> str:
    """
    提取英文、数字和技术名称，构造 PostgreSQL OR 查询。
    中文连续文本交给 Trigram 处理。
    """
    tokens = re.findall(
        r"[a-zA-Z][a-zA-Z0-9_.-]*|[0-9]+",
        query,
    )

    tokens = list(dict.fromkeys(tokens))

    if not tokens:
        return ""

    # to_tsquery 的 OR 操作符
    return " | ".join(
        token.replace("'", "''")
        for token in tokens
    )


# =========================================================
# 4. 文档级排名去重
# =========================================================

def deduplicate_documents(results):
    ranked = []

    for result in results:
        document_id = result.document_id

        if document_id not in ranked:
            ranked.append(document_id)

    return ranked


# =========================================================
# 5. 评估指标
# =========================================================

def calculate_metrics(
    ranked_ids,
    relevant,
    hard_negatives,
    top_k,
):
    top_ids = ranked_ids[:top_k]

    relevant_ids = set(relevant)

    hit = bool(set(top_ids) & relevant_ids)

    recalled = sum(
        1 for document_id in top_ids
        if document_id in relevant_ids
    )

    recall = (
        recalled / len(relevant_ids)
        if relevant_ids
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, document_id in enumerate(top_ids, start=1):
        if document_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    def dcg(relevances):
        return sum(
            (2 ** relevance - 1)
            / math.log2(rank + 1)
            for rank, relevance in enumerate(
                relevances,
                start=1,
            )
        )

    actual_relevances = [
        relevant.get(document_id, 0)
        for document_id in top_ids
    ]

    ideal_relevances = sorted(
        relevant.values(),
        reverse=True,
    )[:top_k]

    ideal_dcg = dcg(ideal_relevances)

    ndcg = (
        dcg(actual_relevances) / ideal_dcg
        if ideal_dcg > 0
        else 0.0
    )

    hard_negative_count = sum(
        1 for document_id in top_ids
        if document_id in hard_negatives
    )

    return {
        "hit": float(hit),
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
        "hard_negative_count": hard_negative_count,
    }



# =========================================================
# 6. FTS / Trigram 独立检索
# =========================================================

async def lexical_search(
    session,
    query,
    fingerprint,
    mode,
    candidate_k=CANDIDATE_K,
):
    if mode == "fts":

        fts_query = build_fts_query(query)

        if not fts_query:
            return []

        tsvector = func.to_tsvector(
            "simple",
            RagChunk.content,
        )

        tsquery = func.to_tsquery(
            "simple",
            fts_query,
        )

        score = func.ts_rank_cd(
            tsvector,
            tsquery,
        )

        statement = (
            select(
                RagChunk,
                RagDocument,
                score.label("score"),
            )
            .join(
                RagDocument,
                RagChunk.document_id == RagDocument.id,
            )
            .where(
                tsvector.op("@@")(tsquery),
                RagChunk.space_id == "rag_documents",
                RagChunk.model_fingerprint == fingerprint,
                RagDocument.status == "active",
            )
            .order_by(
                score.desc(),
                RagChunk.id,
            )
            .limit(candidate_k)
        )

    elif mode == "trigram":

        query_expression = literal(query)

        score = func.word_similarity(
            query_expression,
            RagChunk.content,
        )

        similarity_score = func.similarity(
            RagChunk.content,
            query_expression,
        )

        # -------------------------------------------------
        # Trigram 诊断
        # -------------------------------------------------
        # 不使用 <% 阈值过滤，先观察评测集内的真实分数。
        # 只读取本次 V2 评测文档，不让历史文档干扰诊断。
        # -------------------------------------------------

        diagnostic_statement = (
            select(
                RagDocument.title,
                RagDocument.document_id,
                score.label("word_score"),
                similarity_score.label("similarity_score"),
                func.current_setting(
                    "pg_trgm.word_similarity_threshold"
                ).label("threshold"),
            )
            .join(
                RagDocument,
                RagChunk.document_id == RagDocument.id,
            )
            .where(
                RagChunk.space_id == "rag_documents",
                RagChunk.model_fingerprint == fingerprint,
                RagDocument.status == "active",
                RagDocument.source == "retrieval-evaluation-v2",
            )
            .order_by(
                score.desc(),
                RagChunk.id,
            )
            .limit(5)
        )

        diagnostic_rows = (
            await session.execute(diagnostic_statement)
        ).all()

        print("\n" + "=" * 70)
        print(f"[TRIGRAM DIAGNOSTIC] Query: {query}")
        print("=" * 70)

        if diagnostic_rows:

            threshold = float(diagnostic_rows[0].threshold)

            print(
                f"word_similarity_threshold: {threshold:.4f}"
            )

            max_word_score = max(
                float(row.word_score)
                for row in diagnostic_rows
            )

            max_similarity_score = max(
                float(row.similarity_score)
                for row in diagnostic_rows
            )

            print(
                f"Top 5 max word_similarity: "
                f"{max_word_score:.4f}"
            )

            print(
                f"Top 5 max similarity: "
                f"{max_similarity_score:.4f}"
            )

            print(
                f"Threshold reached: "
                f"{max_word_score >= threshold}"
            )

            print("-" * 70)

            for index, row in enumerate(
                diagnostic_rows,
                start=1,
            ):
                word_score = float(row.word_score)
                sim_score = float(row.similarity_score)

                print(
                    f"#{index} "
                    f"word={word_score:.4f} "
                    f"similarity={sim_score:.4f} "
                    f"document_id={row.document_id}"
                )

                print(f"    title: {row.title}")

        else:
            print(
                "[WARNING] 没有找到 V2 评测文档。"
                "请检查 source 是否为 retrieval-evaluation-v2。"
            )

        print("=" * 70)

        # -------------------------------------------------
        # 正式 Trigram 检索
        # -------------------------------------------------

        # query <% content
        # 判断查询是否与较长文本中的某个连续部分相似
        match_condition = query_expression.op("<%")(
            RagChunk.content
        )

        statement = (
            select(
                RagChunk,
                RagDocument,
                score.label("score"),
            )
            .join(
                RagDocument,
                RagChunk.document_id == RagDocument.id,
            )
            .where(
                match_condition,
                RagChunk.space_id == "rag_documents",
                RagChunk.model_fingerprint == fingerprint,
                RagDocument.status == "active",
            )
            .order_by(
                score.desc(),
                RagChunk.id,
            )
            .limit(candidate_k)
        )

    else:
        raise ValueError(
            f"Unsupported lexical search mode: {mode}"
        )

    rows = (
        await session.execute(statement)
    ).all()

    ranked_ids = []

    for chunk, document, score in rows:
        if document.document_id not in ranked_ids:
            ranked_ids.append(document.document_id)

    return ranked_ids


# =========================================================
# 7. 汇总数据结构
# =========================================================

def create_metric_totals():
    return {
        channel: {
            "hit": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "hard_negative_count": 0,
        }
        for channel in (
            "vector",
            "fts",
            "trigram",
            "hybrid",
        )
    }


# =========================================================
# 8. 主评估流程
# =========================================================

async def main():

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION V2")
    print("=" * 70)

    document_ids = {}

    metric_totals = create_metric_totals()

    try:

        async with app.router.lifespan_context(app):

            print("[PASS] Gateway 初始化成功")

            # -------------------------------------------------
            # 创建评估文档
            # -------------------------------------------------

            async with AsyncSessionLocal() as session:

                ingest_service = RagService(
                    session=session,
                    embedding_space_service=(
                        app.state.embedding_space_service
                    ),
                    space_registry=(
                        app.state.embedding_space_registry
                    ),
                )

                for item in DOCUMENTS:

                    document_id = (
                        f"rag-eval-v2-{uuid4().hex[:8]}-"
                        f"{item['key']}"
                    )

                    request = RagDocumentIngestRequest(
                        document_id=document_id,
                        title=item["title"],
                        content=item["content"],
                        source="retrieval-evaluation-v2",
                        metadata={
                            "evaluation": True,
                            "evaluation_version": "v2",
                            "category": item["key"],
                        },
                        chunk_size=800,
                        chunk_overlap=100,
                    )

                    response = await ingest_service.ingest(
                        request
                    )

                    document_ids[item["key"]] = document_id

                    print(
                        f"[PASS] {item['key']}: "
                        f"{response.chunk_count} chunks"
                    )

            print(
                f"\n[INFO] 文档总数: {len(document_ids)}"
            )

            print(
                f"[INFO] 查询总数: {len(EVALUATION_CASES)}"
            )

            # -------------------------------------------------
            # 执行四种检索评估
            # -------------------------------------------------

            async with AsyncSessionLocal() as session:

                vector_service = RagRetrievalService(
                    session=session,
                    embedding_space_service=(
                        app.state.embedding_space_service
                    ),
                    space_registry=(
                        app.state.embedding_space_registry
                    ),
                )

                hybrid_service = RagHybridSearchService(
                    session=session,
                    embedding_space_service=(
                        app.state.embedding_space_service
                    ),
                    space_registry=(
                        app.state.embedding_space_registry
                    ),
                )

                space = (
                    app.state.embedding_space_registry.get(
                        "rag_documents"
                    )
                )

                if space is None:
                    raise RuntimeError(
                        "RAG Embedding Space 不存在"
                    )

                fingerprint = space.fingerprint

                for index, case in enumerate(
                    EVALUATION_CASES,
                    start=1,
                ):

                    request = RagSearchRequest(
                        query=case["query"],
                        top_k=TOP_K,
                    )

                    # Vector
                    vector_response = (
                        await vector_service.search(request)
                    )

                    vector_ids = deduplicate_documents(
                        vector_response.results
                    )

                    # FTS
                    fts_ids = await lexical_search(
                        session=session,
                        query=case["query"],
                        fingerprint=fingerprint,
                        mode="fts",
                    )

                    # Trigram
                    trigram_ids = await lexical_search(
                        session=session,
                        query=case["query"],
                        fingerprint=fingerprint,
                        mode="trigram",
                    )

                    # Hybrid
                    hybrid_response = (
                        await hybrid_service.search(request)
                    )

                    hybrid_ids = deduplicate_documents(
                        hybrid_response.results
                    )

                    results_by_channel = {
                        "vector": vector_ids,
                        "fts": fts_ids,
                        "trigram": trigram_ids,
                        "hybrid": hybrid_ids,
                    }

                    relevant = {
                        document_ids[key]: grade
                        for key, grade
                        in case["relevant"].items()
                    }

                    hard_negative_ids = {
                        document_ids[key]
                        for key in case["hard_negatives"]
                    }

                    print("\n" + "-" * 70)
                    print(
                        f"[CASE {index}] "
                        f"{case['category']}"
                    )

                    print(
                        f"Query: {case['query']}"
                    )

                    for channel, ranked_ids in (
                        results_by_channel.items()
                    ):

                        metrics = calculate_metrics(
                            ranked_ids=ranked_ids,
                            relevant=relevant,
                            hard_negatives=hard_negative_ids,
                            top_k=TOP_K,
                        )

                        for metric_name, value in (
                            metrics.items()
                        ):
                            metric_totals[channel][
                                metric_name
                            ] += value

                        print(
                            f"\n[{channel.upper()}] "
                            f"Hit={metrics['hit']:.0f} "
                            f"Recall={metrics['recall']:.3f} "
                            f"MRR={metrics['mrr']:.3f} "
                            f"NDCG={metrics['ndcg']:.3f} "
                            f"HardNeg="
                            f"{metrics['hard_negative_count']}"
                        )

                        for rank, doc_id in enumerate(
                            ranked_ids[:TOP_K],
                            start=1,
                        ):

                            key = next(
                                (
                                    key
                                    for key, value
                                    in document_ids.items()
                                    if value == doc_id
                                ),
                                "unknown",
                            )

                            relevance = relevant.get(
                                doc_id,
                                0,
                            )

                            negative = (
                                doc_id in hard_negative_ids
                            )

                            print(
                                f"  {rank}. {key} "
                                f"| relevance={relevance} "
                                f"| hard_negative={negative}"
                            )

                    # Hybrid 通道明细
                    print(
                        "\n[HYBRID CHANNEL DETAILS]"
                    )

                    for result in hybrid_response.results:

                        print(
                            f"  {result.title} "
                            f"| RRF={result.rrf_score} "
                            f"| VectorRank={result.vector_rank} "
                            f"| KeywordRank={result.keyword_rank} "
                            f"| Matched={result.matched_by}"
                        )

            # -------------------------------------------------
            # 汇总
            # -------------------------------------------------

            total = len(EVALUATION_CASES)

            print("\n" + "=" * 70)
            print("RAG EVALUATION V2 SUMMARY")
            print("=" * 70)

            print(
                f"{'Channel':<12}"
                f"{'Hit@5':>10}"
                f"{'Recall@5':>12}"
                f"{'MRR@5':>10}"
                f"{'NDCG@5':>10}"
                f"{'HardNeg':>12}"
            )

            for channel, values in metric_totals.items():

                print(
                    f"{channel:<12}"
                    f"{values['hit'] / total:>10.4f}"
                    f"{values['recall'] / total:>12.4f}"
                    f"{values['mrr'] / total:>10.4f}"
                    f"{values['ndcg'] / total:>10.4f}"
                    f"{values['hard_negative_count'] / total:>12.2f}"
                )

            print("=" * 70)

    finally:

        # -------------------------------------------------
        # 清理评估文档
        # -------------------------------------------------

        if document_ids:

            print("\n[CLEANUP] 清理评估文档")

            async with AsyncSessionLocal() as session:

                cleanup_service = RagService(
                    session=session,
                    embedding_space_service=(
                        app.state.embedding_space_service
                    ),
                    space_registry=(
                        app.state.embedding_space_registry
                    ),
                )

                for document_id in document_ids.values():

                    await cleanup_service.delete(
                        document_id
                    )

            print(
                "[PASS] 评估数据清理完成"
            )


if __name__ == "__main__":
    asyncio.run(main())
