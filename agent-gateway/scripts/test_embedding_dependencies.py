
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.embedding.dependencies import get_embedding_service
from app.embedding.service import EmbeddingService
from app.main import app


def test_dependency_returns_same_instance():
    """验证依赖函数返回应用级单例。"""

    with TestClient(app):

        service1 = get_embedding_service(
            request=_create_request(app)
        )

        service2 = get_embedding_service(
            request=_create_request(app)
        )

        assert isinstance(service1, EmbeddingService)
        assert service1 is service2

        print("[PASS] 获取到 EmbeddingService")
        print("[PASS] 多次获取返回同一个实例")


def test_dependency_not_initialized():
    """验证服务未初始化时返回 HTTP 503。"""

    test_app = FastAPI()

    @test_app.get("/test")
    async def test_endpoint(
        service: EmbeddingService = __import__(
            "fastapi"
        ).Depends(get_embedding_service),
    ):
        return {"success": True}

    with TestClient(test_app) as client:
        response = client.get("/test")

        assert response.status_code == 503

        print("[PASS] 未初始化时返回 HTTP 503")


def _create_request(target_app):
    """构造用于直接调用依赖函数的 Request。"""

    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {
            "version": "3.0",
            "spec_version": "2.3",
        },
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "app": target_app,
    }

    return Request(scope)


def main():
    print("\n========== EMBEDDING DEPENDENCY TEST ==========")

    test_dependency_returns_same_instance()
    test_dependency_not_initialized()

    print("===============================================")
    print("Result: ALL PASSED")


if __name__ == "__main__":
    main()
