
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.spaces.registry import EmbeddingSpaceRegistry
from app.embedding.spaces.schemas import EmbeddingSpace


def register_default_spaces(
    space_registry: EmbeddingSpaceRegistry,
    provider_registry: EmbeddingProviderRegistry,
    provider_name: str,
) -> None:
    """
    注册系统默认的业务 Embedding Spaces。

    当前默认包含：
    1. RAG Documents
    2. Tool Discovery

    两个 Space 共用指定 Provider，
    但拥有独立的 Space ID 和索引名称。
    """

    if not provider_registry.has(provider_name):
        raise ValueError(
            f"Default Embedding Provider not found: {provider_name}"
        )

    provider = provider_registry.get(provider_name)

    # 从真实 Provider 获取模型元数据
    common_metadata = {
        "provider_name": provider.provider_name,
        "model_name": provider.model_name,
        "model_version": provider.model_version,
        "dimension": provider.dimension,
        "fingerprint": provider.fingerprint,
    }

    default_spaces = [
        EmbeddingSpace(
            space_id="rag_documents",
            business_name="RAG",
            description="RAG 文档向量检索空间",
            index_name="rag_documents_v1",
            **common_metadata,
        ),
        EmbeddingSpace(
            space_id="tool_discovery",
            business_name="Tool Discovery",
            description="MCP 工具语义检索空间",
            index_name="tool_discovery_v1",
            **common_metadata,
        ),
    ]

    for space in default_spaces:
        space_registry.register(space)
