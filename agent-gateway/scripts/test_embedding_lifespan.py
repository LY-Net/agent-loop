
from fastapi.testclient import TestClient

from app.main import app


def main():
    print("\n========== EMBEDDING LIFESPAN TEST ==========")

    provider = None

    # 进入 TestClient 上下文时触发 FastAPI startup
    with TestClient(app) as client:

        # 1. 验证 EmbeddingService 初始化
        service = app.state.embedding_service

        assert service is not None

        print("[PASS] EmbeddingService 初始化成功")

        # 2. 验证 Registry 初始化
        registry = app.state.embedding_registry

        assert registry is not None
        assert registry.has("local_bge")

        print("[PASS] EmbeddingProviderRegistry 初始化成功")
        print("[PASS] LocalBGEProvider 注册成功")

        # 3. 获取 Provider，供关闭后检查资源状态
        provider = registry.get("local_bge")

        assert not provider._client.is_closed

        print("[PASS] HTTP 客户端已创建")

        # 4. 验证健康检查接口
        response = client.get("/health")

        assert response.status_code == 200

        result = response.json()

        assert result["success"] is True
        assert result["status"] == "healthy"

        print("[PASS] Gateway Health API 正常")

    # 离开 TestClient 上下文时触发 FastAPI shutdown

    assert provider is not None
    assert provider._client.is_closed

    print("[PASS] HTTP 客户端已正确关闭")

    assert app.state.embedding_service is None
    assert app.state.embedding_registry is None

    print("[PASS] App State 资源已清理")

    print("============================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
