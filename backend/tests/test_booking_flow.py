"""tests/test_booking_flow.py"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def test_user_id():
    return 1

def test_booking_create_valid_train(test_user_id):
    # Train 12627 (Karnataka Express) is in the DB
    payload = {
        "source": "SBC",
        "destination": "NDLS",
        "travel_date": "2026-10-15",
        "travel_class": "3A",
        "passenger_count": 2,
        "train_number": "12627",
        "user_id": test_user_id
    }
    response = client.post("/api/v1/bookings/confirm", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["booking"]["train_number"] == "12627"
    assert data["booking"]["travel_class"] == "3A"
    return data["booking"]["id"]

def test_booking_create_invalid_train(test_user_id):
    payload = {
        "source": "SBC",
        "destination": "NDLS",
        "travel_date": "2026-10-15",
        "travel_class": "3A",
        "passenger_count": 2,
        "train_selection": "99999", # Invalid train
        "user_id": test_user_id
    }
    response = client.post("/api/v1/bookings/confirm", json=payload)
    # The new error handling returns 400 when train matching fails
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False

def test_get_user_bookings(test_user_id):
    response = client.get(f"/api/v1/bookings?user_id={test_user_id}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_cancel_booking():
    # Create one first
    payload = {
        "source": "SBC",
        "destination": "MYS",
        "travel_date": "2026-10-15",
        "travel_class": "CC",
        "passenger_count": 1,
        "train_number": "12007", # Shatabdi
        "user_id": 1
    }
    create_resp = client.post("/api/v1/bookings/confirm", json=payload)
    assert create_resp.status_code == 201
    booking_id = create_resp.json()["booking"]["id"]

    # Cancel it
    cancel_resp = client.delete(f"/api/v1/bookings/{booking_id}")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # Verify status
    get_resp = client.get(f"/api/v1/bookings/{booking_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "CANCELLED"

def test_cancel_invalid_booking():
    response = client.delete("/api/v1/bookings/999999")
    assert response.status_code == 404
