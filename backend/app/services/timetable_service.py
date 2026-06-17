from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from app.models.train_models import Route as RouteModel, Train
from app.repository.route_repo import RouteRepository
from app.repository.station_repo import StationRepository
from app.repository.train_repo import TrainRepository


@dataclass
class RankedTrain:
    train: Any
    score: float
    reasons: List[str]


class TimetableService:
    """Search and ranking layer over train, route, and station repositories."""

    def __init__(self) -> None:
        self.train_repo = TrainRepository()
        self.station_repo = StationRepository()
        self.route_repo = RouteRepository()

    def _expand_codes(self, code_or_city: str, db: Session) -> List[str]:
        if not code_or_city:
            return []
        codes = self.station_repo.get_all_codes_for_city(code_or_city, db)
        if not codes:
            resolved = self.station_repo.fuzzy_find_station(code_or_city, db)
            if resolved:
                codes = [resolved]
        normalized: List[str] = []
        for code in codes:
            if code and code.upper() not in normalized:
                normalized.append(code.upper())
        if code_or_city.upper() not in normalized:
            normalized.append(code_or_city.upper())
        return normalized

    def _parse_time(self, value: Optional[str]) -> Optional[Tuple[int, int]]:
        if not value:
            return None
        text = str(value).strip().lower()
        if text in {"morning", "afternoon", "evening", "night"}:
            ranges = {
                "morning": (5, 0),
                "afternoon": (12, 0),
                "evening": (17, 0),
                "night": (20, 0),
            }
            return ranges[text]
        if len(text) == 5 and text[2] == ":":
            try:
                hh, mm = text.split(":")
                return int(hh), int(mm)
            except Exception:
                return None
        if text.isdigit() and len(text) in {3, 4}:
            if len(text) == 3:
                text = f"0{text}"
            return int(text[:2]), int(text[2:])
        return None

    def _time_to_minutes(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text in {"--", "NA", "N/A"}:
            return None
        if ":" in text:
            try:
                hh, mm = text.split(":")[:2]
                return int(hh) * 60 + int(mm)
            except Exception:
                return None
        if len(text) == 4 and text.isdigit():
            return int(text[:2]) * 60 + int(text[2:])
        return None

    def _train_departure_minutes(self, train: Train, source_code: str, db: Session) -> Optional[int]:
        for attr in ("departure_time", "dep_time", "start_time"):
            val = getattr(train, attr, None)
            m = self._time_to_minutes(val)
            if m is not None:
                return m
        tn = getattr(train, "train_number", None)
        if tn:
            dep = self.route_repo.get_departure_time(tn, source_code, db)
            return self._time_to_minutes(dep)
        return None

    def _train_arrival_minutes(self, train: Train, dest_code: str, db: Session) -> Optional[int]:
        for attr in ("arrival_time", "arr_time", "end_time"):
            val = getattr(train, attr, None)
            m = self._time_to_minutes(val)
            if m is not None:
                return m
        tn = getattr(train, "train_number", None)
        if tn:
            arr = self.route_repo.get_arrival_time(tn, dest_code, db)
            return self._time_to_minutes(arr)
        return None

    def _matches_time_window(
        self,
        train: Train,
        src_code: str,
        db: Session,
        departure_after: Optional[str] = None,
        departure_before: Optional[str] = None,
        time_hint: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        dep_minutes = self._train_departure_minutes(train, src_code, db)
        if dep_minutes is None:
            return True, "departure time unavailable"

        after = self._parse_time(departure_after)
        before = self._parse_time(departure_before)
        if time_hint and not after and not before:
            hint = time_hint.lower()
            if hint == "morning":
                after, before = (5, 0), (11, 59)
            elif hint == "afternoon":
                after, before = (12, 0), (16, 59)
            elif hint == "evening":
                after, before = (17, 0), (21, 59)
            elif hint == "night":
                after = (20, 0)
                before = None

        if after:
            after_m = after[0] * 60 + after[1]
            if dep_minutes < after_m:
                return False, None
        if before:
            before_m = before[0] * 60 + before[1]
            if dep_minutes > before_m:
                return False, None
        return True, None

    def _apply_time_filter(
        self,
        trains: Sequence[Train],
        src_code: str,
        db: Session,
        departure_after: Optional[str] = None,
        departure_before: Optional[str] = None,
        time_hint: Optional[str] = None,
    ) -> List[Train]:
        filtered: List[Train] = []
        unknown: List[Train] = []
        for train in trains:
            ok, note = self._matches_time_window(train, src_code, db, departure_after, departure_before, time_hint)
            if ok:
                if note:
                    unknown.append(train)
                else:
                    filtered.append(train)
        return filtered or unknown or list(trains)

    def search(
        self,
        src: str,
        dst: str,
        date: Optional[str],
        db: Session,
        departure_after: Optional[str] = None,
        departure_before: Optional[str] = None,
        time_hint: Optional[str] = None,
        direct_only: bool = False,
        limit: int = 10,
    ) -> List[Train]:
        src_codes = self._expand_codes(src, db)
        dst_codes = self._expand_codes(dst, db)
        if not src_codes or not dst_codes:
            return []

        results: List[Train] = []
        seen: set = set()

        try:
            direct = (
                db.query(Train)
                .filter(
                    Train.source_station_code.in_(src_codes),
                    Train.destination_station_code.in_(dst_codes),
                )
                .all()
            )
            for train in direct:
                tn = getattr(train, "train_number", None)
                if tn and tn not in seen:
                    results.append(train)
                    seen.add(tn)
        except Exception:
            pass

        if not direct_only:
            try:
                subq_src = (
                    db.query(RouteModel.train_number, RouteModel.sequence)
                    .filter(RouteModel.station_code.in_(src_codes))
                    .subquery()
                )
                subq_dst = (
                    db.query(RouteModel.train_number, RouteModel.sequence)
                    .filter(RouteModel.station_code.in_(dst_codes))
                    .subquery()
                )
                route_trains = (
                    db.query(Train)
                    .join(subq_src, Train.train_number == subq_src.c.train_number)
                    .join(subq_dst, Train.train_number == subq_dst.c.train_number)
                    .filter(subq_src.c.sequence < subq_dst.c.sequence)
                    .all()
                )
                for train in route_trains:
                    tn = getattr(train, "train_number", None)
                    if tn and tn not in seen:
                        results.append(train)
                        seen.add(tn)
            except Exception:
                pass

        if not results:
            for method_name in ("search_between", "search_by_route", "find_between", "get_trains_between"):
                method = getattr(self.train_repo, method_name, None)
                if callable(method):
                    try:
                        fallback = method(src_codes, dst_codes, db)
                        for train in fallback or []:
                            tn = getattr(train, "train_number", None)
                            if tn and tn not in seen:
                                results.append(train)
                                seen.add(tn)
                        if results:
                            break
                    except Exception:
                        continue

        results = self._apply_time_filter(results, src_codes[0], db, departure_after, departure_before, time_hint)
        return results[: max(1, min(int(limit or 10), 50))]

    def search_direct(self, src: str, dst: str, date: Optional[str], db: Session, limit: int = 10) -> List[Train]:
        return self.search(src, dst, date, db, direct_only=True, limit=limit)

    def search_through_trains(self, src: str, dst: str, db: Session, limit: int = 10) -> List[Train]:
        return self.search(src, dst, None, db, direct_only=False, limit=limit)

    def search_ranked(
        self,
        src: str,
        dst: str,
        db: Session,
        departure_after: Optional[str] = None,
        departure_before: Optional[str] = None,
        time_hint: Optional[str] = None,
        limit: int = 10,
    ) -> List[RankedTrain]:
        trains = self.search(src, dst, None, db, departure_after, departure_before, time_hint, False, limit=limit)
        ranked: List[RankedTrain] = []
        for idx, train in enumerate(trains):
            reasons = []
            dep = self._train_departure_minutes(train, self._expand_codes(src, db)[0], db)
            if dep is not None:
                reasons.append(f"departure={dep//60:02d}:{dep%60:02d}")
            score = max(0.0, 100.0 - idx * 5.0)
            ranked.append(RankedTrain(train=train, score=score, reasons=reasons))
        return ranked
