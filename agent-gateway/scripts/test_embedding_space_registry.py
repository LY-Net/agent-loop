
from collections.abc import Mapping
from typing import Any

from app.embedding.providers.base import EmbeddingProvider
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.spaces.schemas import EmbeddingSpace
from app.embedding.spaces.registry import EmbeddingSpaceRegistry


class MockEmbeddingProvider(EmbeddingProvider):

    def __init__(
        self,
        name="local_bge",
        model="BAAI/bge-m3",
        version="v1",
        dimension=1024,
    ):
        self._name = name
        self._model = model
        self._version = version
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_remote(self) -> bool:
        return False

    @property
    def normalize_embeddings(self) -> bool:
        return True

    @property
    def runtime_config(self) -> Mapping[str, Any]:
        return {
            "max_length": 512,
            "device": "cpu",
            "use_fp16": False,
        }

    async def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]

    async def close(self):
        pass


def create_space(
    space_id,
    business_name,
    index_name,
    fingerprint,
    model_name="BAAI/bge-m3",
    model_version="v1",
    dimension=1024,
    provider_name="local_bge",
):
    return EmbeddingSpace(
        space_id=space_id,
        business_name=business_name,
        description=f"{business_name} 语义检索",
        provider_name=provider_name,
        model_name=model_name,
        model_version=model_version,
        dimension=dimension,
        fingerprint=fingerprint,
        index_name=index_name,
    )


def main():
    print("\n========== EMBEDDING SPACE REGISTRY TEST ==========")

    provider_registry = EmbeddingProviderRegistry()

    provider = MockEmbeddingProvider()
    provider_registry.register(provider)

    registry = EmbeddingSpaceRegistry(
        provider_registry=provider_registry
    )

    valid_fingerprint = provider.fingerprint

    rag_space = create_space(
        "rag_documents",
        "RAG",
        "rag_documents_v1",
        valid_fingerprint,
    )

    tool_space = create_space(
        "tool_discovery",
        "Tool Discovery",
        "tool_discovery_v1",
        valid_fingerprint,
    )

    # 1. 正常注册
    registry.register(rag_space)
    registry.register(tool_space)

    assert registry.count == 2
    print("[PASS] 两个业务 Space 注册成功")

    # 2. Space 存在性
    assert registry.has("rag_documents")
    assert registry.has("tool_discovery")

    print("[PASS] Space 存在性查询成功")

    # 3. Space 获取
    result = registry.get("rag_documents")

    assert result.business_name == "RAG"
    assert result.fingerprint == valid_fingerprint

    print("[PASS] Space 获取成功")

    # 4. 重复注册
    try:
        registry.register(rag_space)
        raise AssertionError("重复注册应该失败")
    except ValueError:
        print("[PASS] 重复注册被拒绝")

    # 5. 不存在的 Provider
    try:
        registry.register(
            create_space(
                "invalid_provider",
                "Invalid",
                "invalid_provider_index",
                valid_fingerprint,
                provider_name="unknown_provider",
            )
        )
        raise AssertionError("不存在的 Provider 应该失败")
    except ValueError:
        print("[PASS] 不存在的 Provider 被拒绝")

    # 6. 模型名称不一致
    try:
        registry.register(
            create_space(
                "invalid_model",
                "Invalid",
                "invalid_model_index",
                valid_fingerprint,
                model_name="another-model",
            )
        )
        raise AssertionError("模型不一致应该失败")
    except ValueError:
        print("[PASS] 模型名称不一致被拒绝")

    # 7. 模型版本不一致
    try:
        registry.register(
            create_space(
                "invalid_version",
                "Invalid",
                "invalid_version_index",
                valid_fingerprint,
                model_version="v2",
            )
        )
        raise AssertionError("模型版本不一致应该失败")
    except ValueError:
        print("[PASS] 模型版本不一致被拒绝")

    # 8. 向量维度不一致
    try:
        registry.register(
            create_space(
                "invalid_dimension",
                "Invalid",
                "invalid_dimension_index",
                valid_fingerprint,
                dimension=768,
            )
        )
        raise AssertionError("维度不一致应该失败")
    except ValueError:
        print("[PASS] 向量维度不一致被拒绝")

    # 9. 指纹不一致
    try:
        registry.register(
            create_space(
                "invalid_fingerprint",
                "Invalid",
                "invalid_fingerprint_index",
                "a" * 64,
            )
        )
        raise AssertionError("指纹不一致应该失败")
    except ValueError:
        print("[PASS] 模型指纹不一致被拒绝")

    # 10. 注销
    registry.unregister("tool_discovery")

    assert registry.count == 1
    assert registry.space_ids == ("rag_documents",)

    print("[PASS] Space 注销成功")

    print("===================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
