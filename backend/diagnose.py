#!/usr/bin/env python
"""
Diagnostic script for RailMitra backend.
Run this to test database connectivity, station resolution, train search, and LLM API.
"""
import os
import sys
import json
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure the backend package is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.config import settings
from app.repository.station_repo import StationRepository
from app.services.timetable_service import TimetableService
from app.agent.tools import AgentTools
from app.agent.agent_service import AgentService
from app.core.logger import logger

def test_database():
    print("\n" + "="*60)
    print("TEST 1: Database Connection")
    print("="*60)
    try:
        engine = create_engine(settings.DATABASE_URL)
        with engine.connect() as conn:
            # Use text() for raw SQL
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful.")
            # Count trains
            result = conn.execute(text("SELECT COUNT(*) FROM trains"))
            count = result.scalar()
            print(f"   Number of trains: {count}")
            result = conn.execute(text("SELECT COUNT(*) FROM routes"))
            print(f"   Number of route entries: {result.scalar()}")
            return engine
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def test_station_resolution(engine):
    print("\n" + "="*60)
    print("TEST 2: Station Resolution")
    print("="*60)
    if engine is None:
        print("❌ Skipping: no database connection.")
        return
    Session = sessionmaker(bind=engine)
    db = Session()
    repo = StationRepository()
    stations = ["Kolkata", "Varanasi", "Delhi", "Chennai", "Mumbai", "Bangalore", "Hyderabad"]
    success = True
    for name in stations:
        try:
            code = repo.fuzzy_find_station(name, db)
            if code:
                print(f"   ✅ {name} → {code}")
            else:
                print(f"   ❌ {name} → None (resolution failed)")
                success = False
        except Exception as e:
            print(f"   ❌ {name} → error: {e}")
            success = False
    db.close()
    return success

def test_train_search(engine):
    print("\n" + "="*60)
    print("TEST 3: Train Search (Kolkata → Varanasi)")
    print("="*60)
    if engine is None:
        print("❌ Skipping: no database connection.")
        return
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        timetable = TimetableService()
        trains = timetable.search("Kolkata", "Varanasi", None, db, limit=5)
        if trains:
            print(f"✅ Found {len(trains)} trains:")
            for t in trains[:5]:
                print(f"   - {t.train_number}: {t.train_name}")
        else:
            print("❌ No trains found.")
            # Check if station resolution worked
            src_code = timetable.station_repo.fuzzy_find_station("Kolkata", db)
            dst_code = timetable.station_repo.fuzzy_find_station("Varanasi", db)
            print(f"   Resolved source: {src_code}, destination: {dst_code}")
            # Try direct tool call as well
            tools = AgentTools(db)
            result_json = tools.search_trains("Kolkata", "Varanasi", limit=5)
            result = json.loads(result_json)
            if result.get("status") == "ok":
                trains = result.get("trains", [])
                print(f"   Tool returned {len(trains)} trains.")
            else:
                print(f"   Tool error: {result.get('message', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error during train search: {e}")
    db.close()

def test_llm_api():
    print("\n" + "="*60)
    print("TEST 4: Hugging Face LLM API")
    print("="*60)
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN") or ""
    if not token:
        print("❌ No Hugging Face token found in environment.")
        print("   Please set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN.")
        return False
    print(f"   Token present (first 10 chars): {token[:10]}...")
    import requests
    url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
        "max_tokens": 10,
        "temperature": 0.1,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ LLM API responded: {reply}")
            return True
        else:
            print(f"❌ LLM API error: {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ LLM API request failed: {e}")
        return False

def test_agent_service(engine):
    print("\n" + "="*60)
    print("TEST 5: Agent Service (Local Handler)")
    print("="*60)
    if engine is None:
        print("❌ Skipping: no database connection.")
        return
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        agent = AgentService()
        user_message = "Fare from Kolkata to Varanasi"
        history = []
        response = agent.run(user_message, history, db, session_id="test")
        print(f"Agent response: {response[:200]}...")
        if "fare" in response.lower() or "estimate" in response.lower():
            print("✅ Agent responded with fare-related content.")
        else:
            print("⚠️ Agent response may not be relevant; check full output above.")
    except Exception as e:
        print(f"❌ Agent run failed: {e}")
    db.close()

if __name__ == "__main__":
    print("Starting diagnostics...")
    engine = test_database()
    test_station_resolution(engine)
    test_train_search(engine)
    llm_ok = test_llm_api()
    test_agent_service(engine)
    print("\n" + "="*60)
    print("Diagnostics complete.")
    if not llm_ok:
        print("⚠️ LLM API failed – the assistant will fall back to local handlers.")
    print("="*60)