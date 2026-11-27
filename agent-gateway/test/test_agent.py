
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_agent_chat():

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "user_id": "user_001",
            "session_id": "session_001",
            "message": "分析销售数据",
            "metadata": {}
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["trace_id"]
    assert data["request_id"]
    assert data["session_id"] == "session_001"


def test_trace_id_is_unique():

    payload = {
        "user_id": "user_001",
        "message": "测试请求"
    }

    response1 = client.post(
        "/api/v1/agent/chat",
        json=payload
    )

    response2 = client.post(
        "/api/v1/agent/chat",
        json=payload
    )

    assert (
        response1.headers["X-Trace-ID"]
        != response2.headers["X-Trace-ID"]
    )
