
from fastapi import HTTPException
from starlette.requests import Request
from fastapi.testclient import TestClient

from app.main import app
from app.embedding.dependencies import (
    get_embedding_space_registry,
)
from app.embedding.spaces.registry import EmbeddingSpaceRegistry


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
    print("\n========== EMBEDDING SPACE DEPENDENCIES TEST ==========")

    # 1. 未初始化时，应该返回 HTTP 503
    request = build_request()

    try:
        get_embedding_space_registry(request)
        raise AssertionError("Expected HTTPException was not raised")

    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail == "Embedding Space Registry is not initialized"

    print("[PASS] 未初始化时正确返回 HTTP 503")

    # 2. 启动 Gateway，验证依赖注入
    with TestClient(app):

        expected_registry = app.state.embedding_space_registry

        assert isinstance(
            expected_registry,
            EmbeddingSpaceRegistry,
        )

        actual_registry = get_embedding_space_registry(
            build_request()
        )

        assert actual_registry is expected_registry

        print("[PASS] 成功获取 Embedding Space Registry")
        print("[PASS] 依赖注入返回 app.state 中的同一实例")

        # 3. 验证业务 Space 可正常读取
        assert actual_registry.has("rag_documents")
        assert actual_registry.has("tool_discovery")

        print("[PASS] RAG 和 Tool Discovery Space 均可访问")

    print("========================================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
