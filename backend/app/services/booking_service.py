"""
booking_service.py — Booking orchestration for RailMitra.

This version is backward-compatible with the existing repository/service layout
while adding stronger validation, station resolution, and time-aware fallback
selection when Datameet data is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from app.models import schemas
from app.repository.booking_repo import BookingRepository
from app.repository.station_repo import StationRepository
from app.services.timetable_service import TimetableService


_WORD_TO_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


_CLASS_ALIASES = {
    "general": "GN",
    "unreserved": "GN",
    "gn": "GN",
    "second sitting": "2S",
    "2s": "2S",
    "sitting": "2S",
    "sleeper": "SL",
    "sl": "SL",
    "chair car": "CC",
    "cc": "CC",
    "3ac": "3A",
    "3a": "3A",
    "third ac": "3A",
    "2ac": "2A",
    "2a": "2A",
    "second ac": "2A",
    "1ac": "1A",
    "1a": "1A",
    "first ac": "1A",
    "executive": "EC",
    "ec": "EC",
}


@dataclass
class BookingContext:
    source_station: Optional[str] = None
    destination_station: Optional[str] = None
    date: Optional[str] = None
    train_number: Optional[str] = None
    travel_class: Optional[str] = None
    passenger_count: int = 1
    user_id: int = 1
    departure_after: Optional[str] = None
    departure_before: Optional[str] = None
    time_hint: Optional[str] = None


class BookingService:
    def __init__(self) -> None:
        self.booking_repo = BookingRepository()
        self.timetable_svc = TimetableService()
        self.station_repo = StationRepository()

    def _normalize_class(self, travel_class: Optional[str]) -> str:
        if not travel_class:
            return "SL"
        value = str(travel_class).strip().lower()
        if value.upper() == "ALL":
            return "ALL"
        return _CLASS_ALIASES.get(value, value.upper())

    def _normalize_passengers(self, value: Any) -> int:
        if value is None:
            return 1
        if isinstance(value, int):
            return max(1, min(12, value))
        text = str(value).strip().lower()
        if text.isdigit():
            return max(1, min(12, int(text)))
        return max(1, min(12, _WORD_TO_NUM.get(text, 1)))

    def _normalize_date(self, value: Any) -> str:
        if not value:
            return date.today().isoformat()
        text = str(value).strip()
        if text.lower() in {"today", "now"}:
            return date.today().isoformat()
        if text.lower() == "tomorrow":
            return (date.today() + timedelta(days=1)).isoformat()
        try:
            return datetime.fromisoformat(text).date().isoformat()
        except Exception:
            return date.today().isoformat()

    def _normalize_time(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if len(text) == 5 and text[2] == ":":
            return text
        if len(text) == 4 and text.isdigit():
            return f"{text[:2]}:{text[2:]}"
        return text

    def _resolve_station(self, name_or_code: Optional[str], db: Session) -> Optional[str]:
        if not name_or_code:
            return None
        raw = str(name_or_code).strip()
        if not raw:
            return None

        try:
            candidate = raw.upper()
            if len(candidate) <= 5 and candidate.isalpha():
                station = self.station_repo.get_by_code(candidate, db)
                if station:
                    return candidate
        except Exception:
            pass

        try:
            code = self.station_repo.fuzzy_find_station(raw, db)
            if code:
                return str(code).upper()
        except Exception:
            pass

        lowered = raw.lower()
        aliases = {
            "bangalore": "SBC",
            "bengaluru": "SBC",
            "mysore": "MYS",
            "mysuru": "MYS",
            "mangalore": "MAQ",
            "mangaluru": "MAQ",
            "hubli": "UBL",
            "hubballi": "UBL",
            "yesvantpur": "YPR",
            "yeshwanthpur": "YPR",
            "delhi": "NDLS",
            "new delhi": "NDLS",
            "chennai": "MAS",
            "hyderabad": "HYB",
            "secunderabad": "SC",
            "pune": "PUNE",
            "goa": "MAO",
            "udupi": "UD",
            "hassan": "HAS",
        }
        for key, code in aliases.items():
            if key in lowered:
                return code

        return raw.upper()

    def _pick_train(self, trains: Sequence[Any], entities: Dict[str, Any]) -> Optional[Any]:
        if not trains:
            return None
        requested_train = str(entities.get("train_number") or "").strip()
        if requested_train:
            for train in trains:
                if str(getattr(train, "train_number", "")).strip() == requested_train:
                    return train

        def duration_value(train: Any) -> float:
            for key in ("duration_minutes", "duration", "travel_time", "journey_time"):
                v = getattr(train, key, None)
                if v is not None:
                    return self._coerce_minutes(v)
            return 10**9

        def stop_value(train: Any) -> float:
            for key in ("stops", "total_stops", "stop_count"):
                v = getattr(train, key, None)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        pass
            return 10**9

        def direct_bonus(train: Any) -> int:
            for key in ("is_direct", "direct", "non_stop"):
                v = getattr(train, key, None)
                if isinstance(v, bool) and v:
                    return 0
            return 1

        return sorted(list(trains), key=lambda t: (direct_bonus(t), duration_value(t), stop_value(t), str(getattr(t, "train_number", ""))))[0]

    def _coerce_minutes(self, value: Any) -> float:
        if value is None:
            return 10**9
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        if text.isdigit():
            return float(text)
        minutes = 0.0
        if "h" in text or "m" in text:
            import re
            h = re.search(r"(\d+)\s*h", text)
            m = re.search(r"(\d+)\s*m", text)
            if h:
                minutes += int(h.group(1)) * 60
            if m:
                minutes += int(m.group(1))
            if minutes > 0:
                return minutes
        if ":" in text:
            try:
                parts = text.split(":")
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                pass
        return 10**9

    def search_trains(self, entities: dict, db: Session) -> list:
        src_raw = entities.get("source_station")
        dst_raw = entities.get("destination_station")
        if not src_raw or not dst_raw:
            return []
        src = self._resolve_station(src_raw, db) or str(src_raw).upper()
        dst = self._resolve_station(dst_raw, db) or str(dst_raw).upper()
        return self.timetable_svc.search(
            src,
            dst,
            entities.get("date"),
            db,
            departure_after=self._normalize_time(entities.get("departure_after")),
            departure_before=self._normalize_time(entities.get("departure_before")),
            time_hint=entities.get("time_hint"),
            direct_only=bool(entities.get("direct_only", False)),
            limit=int(entities.get("limit") or 10),
        )

    def create_mock_booking(self, entities: dict, db: Session) -> schemas.Booking:
        src_raw = entities.get("source_station")
        dst_raw = entities.get("destination_station")
        if src_raw:
            entities["source_station"] = self._resolve_station(src_raw, db) or src_raw
        if dst_raw:
            entities["destination_station"] = self._resolve_station(dst_raw, db) or dst_raw

        passenger_num = self._normalize_passengers(entities.get("passenger_count", entities.get("passengers", 1)))
        travel_date = self._normalize_date(entities.get("date") or entities.get("travel_date"))

        trains = self.search_trains(entities, db)
        chosen = self._pick_train(trains, entities)
        train_number = getattr(chosen, "train_number", None) if chosen else (entities.get("train_number") or "00000")

        payload = schemas.BookingCreate(
            user_id=int(entities.get("user_id", 1)),
            train_number=train_number,
            passenger_count=passenger_num,
            travel_class=self._normalize_class(entities.get("class_type") or entities.get("travel_class") or "SL"),
            travel_date=travel_date,
        )
        return self.booking_repo.create(payload, db)

    def cancel_booking(self, booking_id: int, db: Session) -> bool:
        return self.booking_repo.cancel(booking_id, db)

    def list_user_bookings(self, user_id: int, db: Session):
        return self.booking_repo.list_by_user(user_id, db)
