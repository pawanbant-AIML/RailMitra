from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date
from app.models import schemas
from app.repository.booking_repo import BookingRepository
from app.services.timetable_service import TimetableService
from app.repository.station_repo import StationRepository


class BookingService:
    def __init__(self):
        self.booking_repo  = BookingRepository()
        self.timetable_svc = TimetableService()
        self.station_repo  = StationRepository()

    # ------------------------------------------------------------------
    def _resolve_station(self, name_or_code: str, db: Session) -> Optional[str]:
        return self.station_repo.fuzzy_find_station(name_or_code, db)

    # ------------------------------------------------------------------
    def search_trains(self, entities: dict, db: Session) -> list:
        src_raw = entities.get("source_station")
        dst_raw = entities.get("destination_station")
        if not src_raw or not dst_raw:
            return []
        src = self._resolve_station(src_raw, db) or src_raw.upper()
        dst = self._resolve_station(dst_raw, db) or dst_raw.upper()
        return self.timetable_svc.search(src, dst, entities.get("date"), db)

    # ------------------------------------------------------------------
    def create_mock_booking(self, entities: dict, db: Session) -> schemas.Booking:
        # ── Resolve station names → codes ──────────────────────────────
        src_raw = entities.get("source_station")
        dst_raw = entities.get("destination_station")
        if src_raw:
            entities["source_station"] = self._resolve_station(src_raw, db) or src_raw
        if dst_raw:
            entities["destination_station"] = self._resolve_station(dst_raw, db) or dst_raw

        # ── Passenger count ────────────────────────────────────────────
        passenger_str = str(entities.get("passenger_count", "1"))
        word_to_num   = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6}
        if passenger_str.isdigit():
            passenger_num = int(passenger_str)
        else:
            passenger_num = word_to_num.get(passenger_str.lower(), 1)

        # ── Find actual train first ─────────────────────────────────────
        trains = self.search_trains(entities, db)
        train_number = (
            trains[0].train_number
            if trains
            else (entities.get("train_number") or "00000")
        )

        # ── Date fallback → today ──────────────────────────────────────
        travel_date = entities.get("date") or date.today().isoformat()

        payload = schemas.BookingCreate(
            user_id       = int(entities.get("user_id", 1)),
            train_number  = train_number,
            passenger_count = passenger_num,
            travel_class  = entities.get("class_type", "SL"),
            travel_date   = travel_date,
        )
        return self.booking_repo.create(payload, db)

    # ------------------------------------------------------------------
    def cancel_booking(self, booking_id: int, db: Session) -> bool:
        return self.booking_repo.cancel(booking_id, db)

    def list_user_bookings(self, user_id: int, db: Session):
        return self.booking_repo.list_by_user(user_id, db)