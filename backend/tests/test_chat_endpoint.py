"""
tests/test_chat_endpoint.py – Chat endpoint request shape and backward-compatibility tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

import app.api.v1.endpoints.chat as chat_module
from app.main import app


def test_chat_endpoint_accepts_new_payload(monkeypatch):
    def fake_run(user_message, conversation_history, db, session_id):
        assert user_message == "Which is cheapest?"
        assert session_id == "abc123"
        assert conversation_history == [{"role": "user", "content": "hello"}]
        return "Fake reply"

    monkeypatch.setattr(chat_module._agent_svc, "run", fake_run)
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "message": "Which is cheapest?",
            "session_id": "abc123",
            "history": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "Which is cheapest?"},
        {"role": "assistant", "content": "Fake reply"},
    ]


def test_chat_endpoint_accepts_legacy_history_array(monkeypatch):
    def fake_run(user_message, conversation_history, db, session_id):
        assert user_message == "Which is cheapest?"
        assert session_id == "default"
        assert conversation_history == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        return "Legacy reply"

    monkeypatch.setattr(chat_module._agent_svc, "run", fake_run)
    client = TestClient(app)

    payload = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Which is cheapest?"},
    ]

    response = client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    assert response.json() == payload + [{"role": "assistant", "content": "Legacy reply"}]
