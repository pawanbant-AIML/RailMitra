# tests/unit/test_api.py
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_list_trains():
    """Should return a list (possibly empty)"""
    response = client.get("/api/v1/trains")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_chat_empty_message():
    """Sending empty message list should raise 400"""
    response = client.post("/api/v1/chat", json=[])
    assert response.status_code == 400

def test_chat_last_message_not_user():
    """If last message is not from user, should raise 400"""
    payload = [{"role": "assistant", "content": "Hello"}]
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 400