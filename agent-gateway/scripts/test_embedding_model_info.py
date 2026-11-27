
from app.embedding.providers.local_bge import LocalBGEProvider
from app.embedding.registry import EmbeddingProviderRegistry
from app.embedding.service import EmbeddingService


def main():
    print("\n========== EMBEDDING MODEL INFO TEST ==========")

    registry = EmbeddingProviderRegistry()
    provider = LocalBGEProvider()

    registry.register(provider)

    service = EmbeddingService(
        registry=registry,
        provider_name="local_bge",
    )

    try:
        # 1. 获取完整模型元数据
        model_info = service.model_info

        assert model_info.provider_name == "local_bge"
        print("[PASS] Provider 名称正确")

        assert model_info.model_name == "BAAI/bge-m3"
        print("[PASS] 模型名称正确")

        assert model_info.model_version == "v1"
        print("[PASS] 模型版本正确")

        assert model_info.dimension == 1024
        print("[PASS] 向量维度正确")

        assert model_info.is_remote is False
        print("[PASS] Provider 类型正确")

        assert model_info.normalize_embeddings is True
        print("[PASS] 归一化配置正确")

        assert len(model_info.fingerprint) == 64
        print("[PASS] 模型指纹正确")

        # 2. 序列化
        data = model_info.model_dump()

        assert data["model_name"] == "BAAI/bge-m3"
        assert data["fingerprint"] == model_info.fingerprint

        print("[PASS] 模型元数据序列化成功")

        # 3. 不可变性
        try:
            model_info.dimension = 768
            raise AssertionError("模型元数据应该不可修改")
        except Exception as exc:
            if isinstance(exc, AssertionError):
                raise
            print("[PASS] 模型元数据不可修改")

        print(f"Model Fingerprint: {model_info.fingerprint}")

        print("================================================")
        print("Result: ALL PASSED")

    finally:
        import asyncio
        asyncio.run(provider.close())


if __name__ == "__main__":
    main()
