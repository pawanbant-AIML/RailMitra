"""
tests/test_chat_endpoint.py – Chat endpoint request shape and backward-compatibility tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

import app.api.v1.endpoints.chat as chat_module
from app.agent.agent_service import AgentRunResult
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


def test_structured_chat_endpoint_returns_booking_action(monkeypatch):
    def fake_run_structured(user_message, conversation_history, db, session_id):
        assert user_message == "book me a ticket from Bangalore to Mumbai"
        assert session_id == "booking-session"
        assert conversation_history == []
        return AgentRunResult(
            answer="I prepared a booking draft.",
            action="open_booking_form",
            booking_draft={
                "source": "SBC",
                "destination": "CSMT",
                "travel_date": None,
                "travel_class": None,
                "passenger_count": None,
                "train_number": None,
                "direct_only": False,
                "ready_for_submit": False,
                "missing_required_fields": [
                    "travel_date",
                    "travel_class",
                    "passenger_count",
                    "train_number",
                ],
            },
            missing_required_fields=[
                "travel_date",
                "travel_class",
                "passenger_count",
                "train_number",
            ],
            diagnostics={
                "intent": "booking_create",
                "route": "booking_draft",
                "llm_attempted": False,
                "llm_used": False,
                "local_handler_used": True,
                "fallback_used": False,
            },
        )

    monkeypatch.setattr(chat_module._agent_svc, "run_structured", fake_run_structured)
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat/structured",
        json={
            "message": "book me a ticket from Bangalore to Mumbai",
            "session_id": "booking-session",
            "history": [],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "open_booking_form"
    assert payload["booking_draft"]["source"] == "SBC"
    assert payload["booking_draft"]["destination"] == "CSMT"
    assert payload["missing_required_fields"] == [
        "travel_date",
        "travel_class",
        "passenger_count",
        "train_number",
    ]
    assert payload["diagnostics"]["route"] == "booking_draft"
