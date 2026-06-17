"""
recommendation_engine.py — train ranking and recommendation logic.

Supports cheapest / fastest / best-balance / overnight style ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.services.fare_calculator import FareCalculator
from app.services.timetable_service import TimetableService


@dataclass
class Recommendation:
    train: Any
    score: float
    reasons: List[str]
    fare_estimate: Optional[float] = None
    duration_minutes: Optional[float] = None
    stops: Optional[float] = None


class RecommendationEngine:
    def __init__(self) -> None:
        self.fare_calc = FareCalculator()
        self.timetable_svc = TimetableService()

    def _get(self, obj: Any, *keys: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            for key in keys:
                if key in obj and obj[key] not in (None, "", [], {}):
                    return obj[key]
            return default
        for key in keys:
            if hasattr(obj, key):
                value = getattr(obj, key)
                if value not in (None, "", [], {}):
                    return value
        return default

    def _minutes_from_time(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text in {"--", "NA", "N/A"}:
            return None
        if len(text) == 5 and text[2] == ":":
            try:
                hh, mm = text.split(":")
                return int(hh) * 60 + int(mm)
            except Exception:
                return None
        if len(text) == 4 and text.isdigit():
            return int(text[:2]) * 60 + int(text[2:])
        return None

    def _duration_minutes(self, train: Any) -> Optional[float]:
        duration = self._get(train, "duration_minutes", "duration", "travel_time", "journey_time")
        if duration is None:
            return None
        if isinstance(duration, (int, float)):
            return float(duration)
        text = str(duration).strip().lower()
        if text.isdigit():
            return float(text)
        minutes = 0.0
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
                hh, mm = text.split(":")[:2]
                return int(hh) * 60 + int(mm)
            except Exception:
                return None
        return None

    def _stop_count(self, train: Any) -> Optional[int]:
        stops = self._get(train, "stops", "total_stops", "stop_count")
        try:
            return int(stops) if stops is not None else None
        except Exception:
            return None

    def _departure_minutes(self, train: Any) -> Optional[int]:
        for key in ("departure", "dep", "departure_time", "start_time"):
            m = self._minutes_from_time(self._get(train, key))
            if m is not None:
                return m
        return None

    def _fare_for_train(self, train: Any, source: str, destination: str, travel_class: str, passengers: int) -> Optional[float]:
        tn = self._get(train, "train_number")
        if not tn:
            return None
        breakdown = self.fare_calc.calculate(
            travel_class=travel_class or "SL",
            distance_km=None,
            train_name=self._get(train, "train_name"),
            passengers=passengers,
            source_code=source,
            dest_code=destination,
        )
        return float(breakdown.total_fare)

    def rank(
        self,
        trains: Sequence[Any],
        source: str,
        destination: str,
        time_hint: Optional[str] = None,
        departure_after: Optional[str] = None,
        departure_before: Optional[str] = None,
        preference: Optional[str] = None,
        sort_by: Optional[str] = None,
        direct_only: bool = False,
        travel_class: Optional[str] = None,
        passengers: int = 1,
        limit: int = 10,
    ) -> List[Recommendation]:
        candidates: List[Recommendation] = []
        for train in trains or []:
            dur = self._duration_minutes(train)
            stops = self._stop_count(train)
            dep = self._departure_minutes(train)
            fare_est = self._fare_for_train(train, source, destination, travel_class or "SL", passengers)
            score = 100.0
            reasons: List[str] = []

            if direct_only and stops is not None:
                score += 20.0 if stops <= 2 else -20.0
                reasons.append("direct only requested")
            if sort_by == "fare" or preference == "low_cost":
                if fare_est is not None:
                    score -= fare_est / 100.0
                    reasons.append("lower fare preferred")
            if sort_by == "duration" or preference == "fastest":
                if dur is not None:
                    score -= dur / 10.0
                    reasons.append("shorter journey preferred")
            if sort_by == "stops":
                if stops is not None:
                    score -= stops * 2.0
                    reasons.append("fewer stops preferred")

            if time_hint:
                hint = time_hint.lower()
                if dep is not None:
                    hh = dep // 60
                    if hint == "morning" and 5 <= hh <= 11:
                        score += 15
                        reasons.append("morning preference matched")
                    elif hint == "afternoon" and 12 <= hh <= 16:
                        score += 15
                        reasons.append("afternoon preference matched")
                    elif hint == "evening" and 17 <= hh <= 21:
                        score += 15
                        reasons.append("evening preference matched")
                    elif hint == "night" and (hh >= 22 or hh <= 4):
                        score += 15
                        reasons.append("overnight preference matched")

            if departure_after and dep is not None:
                try:
                    hh, mm = departure_after.split(":")
                    if dep >= int(hh) * 60 + int(mm):
                        score += 5
                except Exception:
                    pass
            if departure_before and dep is not None:
                try:
                    hh, mm = departure_before.split(":")
                    if dep <= int(hh) * 60 + int(mm):
                        score += 5
                except Exception:
                    pass

            candidates.append(
                Recommendation(
                    train=train,
                    score=score,
                    reasons=reasons,
                    fare_estimate=fare_est,
                    duration_minutes=dur,
                    stops=stops,
                )
            )

        candidates.sort(
            key=lambda r: (
                -r.score,
                r.fare_estimate if r.fare_estimate is not None else 10**9,
                r.duration_minutes if r.duration_minutes is not None else 10**9,
                r.stops if r.stops is not None else 10**9,
                str(self._get(r.train, "train_number", default="")),
            )
        )
        return candidates[: max(1, min(int(limit or 10), 20))]
