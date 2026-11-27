
import asyncio

from app.core.config import settings
from app.embedding.providers.base import EmbeddingProvider
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.schemas import EmbeddingInput
from app.embedding.service import EmbeddingService


class MockEmbeddingProvider(EmbeddingProvider):

    def __init__(self):
        self.calls = []

    @property
    def provider_name(self):
        return "mock_local"

    @property
    def model_name(self):
        return "mock-model"

    @property
    def dimension(self):
        return settings.EMBEDDING_DIMENSION

    @property
    def is_remote(self):
        return False

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * self.dimension for _ in texts]

    async def close(self):
        pass


async def main():
    print("\n========== EMBEDDING INPUT LIMIT TEST ==========")

    registry = EmbeddingProviderRegistry()
    provider = MockEmbeddingProvider()
    registry.register(provider)

    service = EmbeddingService(
        registry=registry,
        provider_name="mock_local",
    )

    try:
        # 1. 超过最大文本数量
        try:
            await service.embed(
                EmbeddingInput(
                    texts=[
                        f"text-{i}"
                        for i in range(settings.EMBEDDING_MAX_TEXTS + 1)
                    ]
                )
            )
            raise AssertionError("超过最大文本数量未被拦截")

        except ValueError:
            assert len(provider.calls) == 0
            print("[PASS] 超过最大文本数量被拦截")

        # 2. 单条文本超过最大长度
        try:
            await service.embed(
                EmbeddingInput(
                    texts=[
                        "a" * (settings.EMBEDDING_MAX_TEXT_LENGTH + 1)
                    ]
                )
            )
            raise AssertionError("超过最大文本长度未被拦截")

        except ValueError:
            assert len(provider.calls) == 0
            print("[PASS] 超过最大文本长度被拦截")

        # 3. 合法输入
        result = await service.embed(
            EmbeddingInput(
                texts=["正常测试文本"]
            )
        )

        assert len(result) == 1
        assert len(result[0]) == settings.EMBEDDING_DIMENSION

        print("[PASS] 合法输入正常生成向量")

        # 4. 验证非法输入没有触发 Provider
        assert len(provider.calls) == 1

        print("[PASS] 非法输入未触发 Provider")

    finally:
        await registry.close_all()

    print("================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
