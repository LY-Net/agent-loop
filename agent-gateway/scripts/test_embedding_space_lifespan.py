
from fastapi.testclient import TestClient

from app.main import app
from app.embedding.spaces.service import EmbeddingSpaceService


def main():
    print("\n========== EMBEDDING SPACE LIFESPAN TEST ==========")

    provider = None

    with TestClient(app) as client:

        # 1. Provider Registry
        provider_registry = app.state.embedding_registry

        assert provider_registry is not None
        assert provider_registry.has("local_bge")

        provider = provider_registry.get("local_bge")

        print("[PASS] Provider Registry 初始化成功")

        # 2. Embedding Service
        embedding_service = app.state.embedding_service

        assert embedding_service is not None
        assert embedding_service.model_info.model_name == "BAAI/bge-m3"

        print("[PASS] Embedding Service 初始化成功")

        # 3. Space Registry
        space_registry = app.state.embedding_space_registry

        assert space_registry is not None
        assert space_registry.count == 2

        print("[PASS] Space Registry 初始化成功")

        # 4. 默认业务 Space
        rag_space = space_registry.get("rag_documents")
        tool_space = space_registry.get("tool_discovery")

        assert rag_space.index_name == "rag_documents_v1"
        assert tool_space.index_name == "tool_discovery_v1"

        assert rag_space.fingerprint == tool_space.fingerprint
        assert rag_space.fingerprint == provider.fingerprint

        print("[PASS] 两个默认 Space 注册成功")
        print("[PASS] 两个 Space 模型配置一致")

        # 5. Embedding Space Service
        space_service = app.state.embedding_space_service

        assert space_service is not None
        assert isinstance(space_service, EmbeddingSpaceService)

        print("[PASS] Embedding Space Service 初始化成功")

        # 6. 检查对象依赖关系
        assert space_service._embedding_service is embedding_service
        assert space_service._space_registry is space_registry

        print("[PASS] Embedding Space Service 依赖关系正确")

        # 7. Health API
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        print("[PASS] Health API 正常")

    # 8. 检查 Provider 资源释放
    assert provider._client.is_closed

    print("[PASS] Provider HTTP Client 已关闭")

    # 9. 检查 app.state 清理
    assert app.state.embedding_registry is None
    assert app.state.embedding_service is None
    assert app.state.embedding_space_registry is None
    assert app.state.embedding_space_service is None

    print("[PASS] FastAPI app.state 已清理")

    print("====================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
