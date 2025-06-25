"""テスト設定."""
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app


@pytest.fixture
def client():
    """テスト用クライアント."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """非同期テスト用クライアント."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
