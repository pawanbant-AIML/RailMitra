import pytest
from sqlalchemy.orm import Session
from app.services.timetable_service import TimetableService
from app.services.booking_service import BookingService
from app.models.db import SessionLocal

@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()

def test_timetable_search(db: Session):
    svc = TimetableService()
    results = svc.search("BENG", "CHN", "2023-12-01", db)
    assert isinstance(results, list)

def test_mock_booking(db: Session):
    svc = BookingService()
    entities = {
        "user_id": 1,
        "train_number": "12687",
        "passenger_count": "2",
        "class_type": "SL",
        "date": "2023-12-25",
    }
    booking = svc.create_mock_booking(entities, db)
    assert booking.id is not None
    assert booking.status == "CONFIRMED"