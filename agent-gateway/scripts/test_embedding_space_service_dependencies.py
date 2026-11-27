
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.main import app
from app.embedding.dependencies import (
    get_embedding_space_service,
)
from app.embedding.spaces.service import EmbeddingSpaceService


def build_request() -> Request:
    return Request(
        {
            "type": "http",
            "app": app,
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
        }
    )


def main():
    print("\n========== EMBEDDING SPACE SERVICE DEPENDENCIES TEST ==========")

    # 1. 未初始化时，应该返回 HTTP 503
    request = build_request()

    try:
        get_embedding_space_service(request)
        raise AssertionError("Expected HTTPException was not raised")

    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail == "Embedding Space Service is not initialized"

    print("[PASS] 未初始化时正确返回 HTTP 503")

    # 2. 启动 Gateway，验证依赖注入
    with TestClient(app):

        expected_service = app.state.embedding_space_service

        assert expected_service is not None
        assert isinstance(
            expected_service,
            EmbeddingSpaceService,
        )

        print("[PASS] Embedding Space Service 初始化成功")

        # 3. 验证依赖注入返回同一个实例
        actual_service = get_embedding_space_service(
            build_request()
        )

        assert actual_service is expected_service

        print("[PASS] 依赖注入返回 app.state 中的同一实例")

        # 4. 验证内部依赖关系
        assert (
            actual_service._embedding_service
            is app.state.embedding_service
        )

        assert (
            actual_service._space_registry
            is app.state.embedding_space_registry
        )

        print("[PASS] Embedding Space Service 内部依赖关系正确")

        # 5. 验证业务 Space
        assert app.state.embedding_space_registry.has(
            "rag_documents"
        )

        assert app.state.embedding_space_registry.has(
            "tool_discovery"
        )

        print("[PASS] RAG 和 Tool Discovery Space 均可访问")

    # 6. 验证生命周期关闭后的状态
    assert app.state.embedding_space_service is None

    print("[PASS] Gateway 关闭后 Space Service 已清理")

    print("==============================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
