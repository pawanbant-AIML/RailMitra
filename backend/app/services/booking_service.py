"""
booking_service.py — Booking orchestration for RailMitra.

This version is backward-compatible with the existing repository/service layout
while adding stronger validation, station resolution, and fallback selection
when Datameet data is incomplete.
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


class BookingService:
    def __init__(self) -> None:
        self.booking_repo = BookingRepository()
        self.timetable_svc = TimetableService()
        self.station_repo = StationRepository()

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

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

    def _resolve_station(self, name_or_code: Optional[str], db: Session) -> Optional[str]:
        if not name_or_code:
            return None

        raw = str(name_or_code).strip()
        if not raw:
            return None

        # Direct code check first.
        try:
            candidate = raw.upper()
            if len(candidate) <= 5 and candidate.isalpha():
                station = self.station_repo.get_by_code(candidate, db)
                if station:
                    return candidate
        except Exception:
            pass

        # Fuzzy search on the repo.
        try:
            code = self.station_repo.fuzzy_find_station(raw, db)
            if code:
                return str(code).upper()
        except Exception:
            pass

        # Lightweight alias fallback.
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
        lowered = raw.lower()
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

        # Prefer direct route, then shortest duration, then low stops.
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

        ranked = sorted(
            list(trains),
            key=lambda t: (direct_bonus(t), duration_value(t), stop_value(t), str(getattr(t, "train_number", ""))),
        )
        return ranked[0] if ranked else None

    def _coerce_minutes(self, value: Any) -> float:
        if value is None:
            return 10**9
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        if text.isdigit():
            return float(text)
        # Parses "5h 40m", "5:40", "340 mins"
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
            parts = text.split(":")
            try:
                return float(int(parts[0]) * 60 + int(parts[1]))
            except Exception:
                pass
        try:
            return float(text)
        except Exception:
            return 10**9

    # ------------------------------------------------------------------
    # Search / booking
    # ------------------------------------------------------------------

    def search_trains(self, entities: dict, db: Session) -> list:
        src_raw = entities.get("source_station")
        dst_raw = entities.get("destination_station")
        if not src_raw or not dst_raw:
            return []

        src = self._resolve_station(src_raw, db) or str(src_raw).upper()
        dst = self._resolve_station(dst_raw, db) or str(dst_raw).upper()
        travel_date = entities.get("date") or entities.get("travel_date")

        trains = self.timetable_svc.search(src, dst, travel_date, db)
        if trains:
            return trains

        # Graceful retry: try with canonical station codes from the same city if available.
        try:
            src_alt = self.station_repo.get_all_codes_for_city(src, db) or [src]
            dst_alt = self.station_repo.get_all_codes_for_city(dst, db) or [dst]
            seen = set()
            collected = []
            for s in src_alt:
                for d in dst_alt:
                    for train in self.timetable_svc.search(s, d, travel_date, db):
                        tn = getattr(train, "train_number", None)
                        if tn and tn not in seen:
                            collected.append(train)
                            seen.add(tn)
            return collected
        except Exception:
            return []

    def create_mock_booking(self, entities: dict, db: Session) -> schemas.Booking:
        """
        Create a booking using the best available train match.

        This is intentionally resilient:
        - resolves station names to codes
        - normalizes passenger counts and classes
        - picks a reasonable train when no explicit train number is supplied
        - falls back to today's date if no date is provided
        """
        src_raw = entities.get("source_station")
        dst_raw = entities.get("destination_station")
        travel_class = self._normalize_class(entities.get("class_type") or entities.get("travel_class"))
        passenger_num = self._normalize_passengers(
            entities.get("passenger_count") or entities.get("passengers") or 1
        )
        travel_date = self._normalize_date(entities.get("date") or entities.get("travel_date"))
        user_id = int(entities.get("user_id", 1) or 1)

        source_code = self._resolve_station(src_raw, db) if src_raw else None
        destination_code = self._resolve_station(dst_raw, db) if dst_raw else None

        if source_code:
            entities["source_station"] = source_code
        if destination_code:
            entities["destination_station"] = destination_code

        trains = self.search_trains(entities, db)
        chosen_train = self._pick_train(trains, entities)

        train_number = (
            str(entities.get("train_number")).strip()
            if entities.get("train_number")
            else None
        )
        if not train_number and chosen_train is not None:
            train_number = str(getattr(chosen_train, "train_number", "")).strip() or None
        if not train_number:
            train_number = "00000"

        payload = schemas.BookingCreate(
            user_id=user_id,
            train_number=train_number,
            passenger_count=passenger_num,
            travel_class=travel_class if travel_class != "ALL" else "SL",
            travel_date=travel_date,
        )
        return self.booking_repo.create(payload, db)

    def cancel_booking(self, booking_id: int, db: Session) -> bool:
        try:
            return bool(self.booking_repo.cancel(int(booking_id), db))
        except Exception:
            return False

    def list_user_bookings(self, user_id: int, db: Session):
        try:
            return self.booking_repo.list_by_user(int(user_id), db)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Optional helpers for future agent wiring
    # ------------------------------------------------------------------

    def find_best_booking_candidate(self, entities: dict, db: Session):
        trains = self.search_trains(entities, db)
        return self._pick_train(trains, entities)

    def modify_booking(self, booking_id: int, updates: dict, db: Session):
        """
        Best-effort booking modification wrapper.
        Uses repo methods if they exist; otherwise returns None.
        """
        repo = self.booking_repo
        for method_name in ("update", "modify", "update_booking", "edit"):
            method = getattr(repo, method_name, None)
            if callable(method):
                try:
                    return method(int(booking_id), updates, db)
                except TypeError:
                    try:
                        return method(int(booking_id), db, updates)
                    except Exception:
                        continue
                except Exception:
                    continue
        return None
