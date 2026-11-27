
import asyncio

from app.embedding.providers.local_bge import LocalBGEProvider


async def main():
    print("\n========== LOCAL BGE FINGERPRINT TEST ==========")

    provider = LocalBGEProvider()

    try:
        # 1. Provider 基础信息
        assert provider.provider_name == "local_bge"
        print("[PASS] Provider 名称正确")

        # 2. 模型名称
        assert provider.model_name == "BAAI/bge-m3"
        print("[PASS] 模型名称正确")

        # 3. 模型版本
        assert provider.model_version == "v1"
        print("[PASS] 模型版本信息正确")

        # 4. 向量维度
        assert provider.dimension == 1024
        print("[PASS] 向量维度正确")

        # 5. 归一化配置
        assert isinstance(provider.normalize_embeddings, bool)
        print("[PASS] 归一化配置存在")

        # 6. 运行配置
        runtime_config = provider.runtime_config

        assert isinstance(runtime_config, dict)
        assert runtime_config["max_length"] == 512
        assert runtime_config["device"] == "cpu"

        print("[PASS] 模型运行配置正确")

        # 7. 指纹生成
        fingerprint = provider.fingerprint

        assert len(fingerprint) == 64
        assert all(
            char in "0123456789abcdef"
            for char in fingerprint
        )

        print("[PASS] SHA256 指纹生成成功")

        # 8. 指纹稳定性
        assert provider.fingerprint == fingerprint
        print("[PASS] 指纹生成结果稳定")

        print(f"Model Fingerprint: {fingerprint}")

        print("================================================")
        print("Result: ALL PASSED")

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
