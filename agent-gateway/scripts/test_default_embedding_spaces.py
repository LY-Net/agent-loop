
import asyncio

from app.embedding.providers.local_bge import LocalBGEProvider
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.spaces.defaults import register_default_spaces
from app.embedding.spaces.registry import EmbeddingSpaceRegistry


async def main():
    print("\n========== DEFAULT EMBEDDING SPACES TEST ==========")

    provider_registry = EmbeddingProviderRegistry()
    provider = LocalBGEProvider()

    provider_registry.register(provider)

    space_registry = EmbeddingSpaceRegistry(
        provider_registry=provider_registry
    )

    try:
        # 1. 注册默认 Space
        register_default_spaces(
            space_registry=space_registry,
            provider_registry=provider_registry,
            provider_name="local_bge",
        )

        assert space_registry.count == 2
        print("[PASS] 两个默认 Space 注册成功")

        # 2. 检查 RAG Space
        rag_space = space_registry.get("rag_documents")

        assert rag_space.business_name == "RAG"
        assert rag_space.index_name == "rag_documents_v1"

        print("[PASS] RAG Space 配置正确")

        # 3. 检查 Tool Discovery Space
        tool_space = space_registry.get("tool_discovery")

        assert tool_space.business_name == "Tool Discovery"
        assert tool_space.index_name == "tool_discovery_v1"

        print("[PASS] Tool Discovery Space 配置正确")

        # 4. 检查两个 Space 使用同一个 Provider
        assert rag_space.provider_name == tool_space.provider_name

        print("[PASS] 两个 Space 共用同一个 Provider")

        # 5. 检查模型指纹一致
        assert rag_space.fingerprint == provider.fingerprint
        assert tool_space.fingerprint == provider.fingerprint

        print("[PASS] 两个 Space 模型指纹一致")

        # 6. 检查向量维度
        assert rag_space.dimension == 1024
        assert tool_space.dimension == 1024

        print("[PASS] 两个 Space 向量维度正确")

        # 7. 检查 Space 独立性
        assert rag_space.space_id != tool_space.space_id
        assert rag_space.index_name != tool_space.index_name

        print("[PASS] 两个 Space 逻辑隔离")

        print("===================================================")
        print("Result: ALL PASSED")

    finally:
        await provider_registry.close_all()


if __name__ == "__main__":
    asyncio.run(main())
