"""
tests/test_agent.py – Integration smoke tests for the RailMitra agent.

Runs against the local SQLite DB (train_ticket.db).
Does NOT call the HuggingFace API – uses the fallback handler only.
Run with: pytest tests/test_agent.py -v
"""

import sys
import os

# Make sure we can import from the backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db import Base
from app.agent.agent_service import AgentService
from app.agent.tools import AgentTools
from app.services.fare_calculator import FareCalculator


# ---------------------------------------------------------------------------
# DB fixture – use the local SQLite file
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "train_ticket.db")

@pytest.fixture(scope="module")
def db():
    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# FareCalculator unit tests
# ---------------------------------------------------------------------------

class TestFareCalculator:
    def setup_method(self):
        self.calc = FareCalculator()

    def test_sleeper_fare_distance(self):
        fb = self.calc.calculate("SL", distance_km=352)
        assert fb.final_fare >= 110, "Sleeper must meet minimum fare floor"
        assert fb.final_fare <= 500, "Sleeper on 352km should be under ₹500"

    def test_class_ordering(self):
        """GN < SL < 3A < 2A < 1A for same distance."""
        fares = self.calc.calculate_all_classes(distance_km=400)
        prices = {cls: fares[cls].final_fare for cls in fares}
        assert prices["GN"] < prices["SL"], "GN must be cheaper than SL"
        assert prices["SL"] < prices["3A"], "SL must be cheaper than 3A"
        assert prices["3A"] < prices["2A"], "3A must be cheaper than 2A"
        assert prices["2A"] < prices["1A"], "2A must be cheaper than 1A"

    def test_rajdhani_multiplier(self):
        base = self.calc.calculate("2A", distance_km=400, train_name="Karnataka Express")
        premium = self.calc.calculate("2A", distance_km=400, train_name="Rajdhani Express")
        assert premium.final_fare > base.final_fare, "Rajdhani must cost more than regular express"

    def test_passenger_scaling(self):
        one = self.calc.calculate("SL", distance_km=352, passengers=1)
        two = self.calc.calculate("SL", distance_km=352, passengers=2)
        assert two.total_fare == one.total_fare * 2

    def test_fallback_when_no_distance(self):
        fb = self.calc.calculate("3A", distance_km=None)
        assert fb.final_fare >= 300, "3A minimum fare floor must hold"
        assert fb.is_estimated is True

    def test_corridor_lookup(self):
        fb = self.calc.calculate("SL", distance_km=None, source_code="SBC", dest_code="MAQ")
        assert fb.distance_km == 352
        assert fb.final_fare >= 110

    def test_all_classes_format(self):
        fares = self.calc.calculate_all_classes(distance_km=300, train_name="Express")
        table = self.calc.format_fare_table("SBC", "MAQ", "16585", "Malabar Express", fares)
        assert "₹" in table
        assert "Sleeper" in table


# ---------------------------------------------------------------------------
# AgentTools integration tests (need real DB)
# ---------------------------------------------------------------------------

class TestAgentTools:
    def test_search_trains_bangalore_mangalore(self, db):
        tools = AgentTools(db)
        import json
        result = json.loads(tools.search_trains("Bangalore", "Mangalore"))
        assert result["status"] in ("ok", "no_results"), f"Unexpected: {result}"

    def test_get_fare_all_classes(self, db):
        tools = AgentTools(db)
        import json
        # First get any train
        trains = json.loads(tools.search_trains("Bangalore", "Mangalore"))
        if trains.get("status") == "ok" and trains["trains"]:
            tn = trains["trains"][0]["train_number"]
            result = json.loads(tools.get_fare(tn, "Bangalore", "Mangalore", "ALL", 1))
            assert result["status"] == "ok"
            assert "fares" in result

    def test_get_train_route(self, db):
        tools = AgentTools(db)
        import json
        trains = json.loads(tools.search_trains("Bangalore", "Mangalore"))
        if trains.get("status") == "ok" and trains["trains"]:
            tn = trains["trains"][0]["train_number"]
            result = json.loads(tools.get_train_route(tn))
            assert result["status"] == "ok"
            assert "route" in result
            assert len(result["route"]) > 0

    def test_booking_flow(self, db):
        tools = AgentTools(db)
        import json
        # Book
        book_result = json.loads(tools.book_ticket("Bangalore", "Mangalore", "SL", 2))
        assert book_result["status"] == "confirmed"
        bid = book_result["booking_id"]
        # History
        hist = json.loads(tools.get_booking_history())
        assert hist["status"] == "ok"
        ids = [b["booking_id"] for b in hist["bookings"]]
        assert bid in ids
        # Cancel
        cancel = json.loads(tools.cancel_booking(bid))
        assert cancel["status"] == "cancelled"

    def test_station_info(self, db):
        tools = AgentTools(db)
        import json
        result = json.loads(tools.get_station_info("SBC"))
        # Either found or not — both are valid statuses
        assert result["status"] in ("ok", "not_found")


# ---------------------------------------------------------------------------
# AgentService fallback handler tests (no LLM required)
# ---------------------------------------------------------------------------

class TestAgentFallback:
    def setup_method(self):
        # Force fallback by clearing the token
        self.svc = AgentService()
        self.svc.hf_token = ""  # disable LLM to test fallback only

    def test_greeting(self, db):
        reply = self.svc.run("Hello!", [], db)
        assert "RailMitra" in reply or "help" in reply.lower()

    def test_search_from_message(self, db):
        reply = self.svc.run("Show trains from Bangalore to Mangalore", [], db)
        assert any(word in reply.lower() for word in ["train", "found", "no trains"])

    def test_fare_query(self, db):
        reply = self.svc.run("What is the sleeper fare from Bangalore to Mangalore?", [], db)
        assert any(c in reply for c in ["₹", "fare", "Sleeper", "no trains", "Please"])

    def test_ambiguous_query(self, db):
        reply = self.svc.run("I need a train", [], db)
        # Should ask for clarification or say it didn't understand
        assert len(reply) > 10

    def test_context_resolution(self, db):
        history = [
            {"role": "user", "content": "Show trains from Bangalore to Mangalore"},
            {"role": "assistant", "content": "Found 3 trains..."},
        ]
        reply = self.svc.run("What is the sleeper fare?", history, db)
        assert len(reply) > 10
