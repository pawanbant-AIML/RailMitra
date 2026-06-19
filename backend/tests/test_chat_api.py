"""tests/test_chat_api.py"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_structured_chat_greeting():
    payload = {
        "message": "Hi",
        "session_id": "test_session_1"
    }
    response = client.post("/api/v1/chat/structured", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    # Greetings usually go to local handler if LLM is disabled
    assert len(data["messages"]) > 0
    assert data["messages"][-1]["role"] == "assistant"
    assert "diagnostics" in data

def test_structured_chat_train_search():
    payload = {
        "message": "Find trains from Bangalore to Mumbai tomorrow",
        "session_id": "test_session_2"
    }
    response = client.post("/api/v1/chat/structured", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Verify the local fallback parsed the intent correctly
    assert data["diagnostics"]["intent"] in ("train_search", "multi_intent")
    
    msg_content = data["messages"][-1]["content"].lower()
    # Should mention the cities or have results
    assert "mumbai" in msg_content or "csmt" in msg_content or "train" in msg_content

def test_structured_chat_booking_intent():
    payload = {
        "message": "Book 2 sleeper tickets from Bangalore to Chennai tomorrow",
        "session_id": "test_session_3"
    }
    response = client.post("/api/v1/chat/structured", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Should detect booking intent and trigger the UI action
    assert data["action"] == "open_booking_form"
    assert "booking_draft" in data
    draft = data["booking_draft"]
    
    assert draft["source"] is not None
    assert draft["destination"] is not None
    assert draft["passenger_count"] == 2
    assert draft["travel_class"] in ("SL", "sleeper")

def test_structured_chat_empty_message():
    payload = {
        "message": "   ",
        "session_id": "test_session_4"
    }
    response = client.post("/api/v1/chat/structured", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "Message cannot be empty" in data["detail"]
