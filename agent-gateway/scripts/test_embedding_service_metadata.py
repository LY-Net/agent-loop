
import asyncio

from app.embedding.providers.base import EmbeddingProvider
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.service import EmbeddingService


class MockEmbeddingProvider(EmbeddingProvider):

    @property
    def provider_name(self):
        return "mock_local"

    @property
    def model_name(self):
        return "mock-model-v1"

    @property
    def dimension(self):
        return 1024

    @property
    def is_remote(self):
        return False

    async def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]

    async def close(self):
        pass


async def main():
    print("\n========== EMBEDDING SERVICE METADATA TEST ==========")

    registry = EmbeddingProviderRegistry()
    registry.register(MockEmbeddingProvider())

    service = EmbeddingService(
        registry=registry,
        provider_name="mock_local",
    )

    try:
        info = service.model_info

        assert info.provider_name == "mock_local"
        assert info.model_name == "mock-model-v1"
        assert info.dimension == 1024
        assert info.is_remote is False

        print("[PASS] Service 获取 Provider 元数据")
        print("[PASS] Provider 名称正确")
        print("[PASS] 模型名称正确")
        print("[PASS] 向量维度正确")
        print("[PASS] 本地 Provider 属性正确")

        print("\nModel Metadata:")
        print(info.model_dump())

    finally:
        await registry.close_all()

    print("=====================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
