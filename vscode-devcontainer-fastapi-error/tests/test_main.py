"""メインアプリケーションのテスト."""
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """ルートエンドポイントのテスト."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_health_check(client: TestClient):
    """ヘルスチェックエンドポイントのテスト."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_openapi_docs(client: TestClient):
    """OpenAPI仕様書のテスト."""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
