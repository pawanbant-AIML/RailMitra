import requests

BASE = "http://localhost:8000/api/v1"

def test_chat_flow():
    payload = [
        {"role": "user", "content": "Find trains from BENG to CHN tomorrow"}
    ]
    r = requests.post(f"{BASE}/chat", json=payload, timeout=10)
    assert r.status_code == 200
    resp = r.json()
    assert resp[-1]["role"] == "assistant"
    assert "train" in resp[-1]["content"].lower()