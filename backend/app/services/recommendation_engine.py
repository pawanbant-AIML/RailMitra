"""
recommendation_engine.py — train ranking and recommendation logic.

This module is intentionally DB-light and works on either ORM objects or dicts.
It can be plugged into the agent later to answer:
- cheapest train
- fastest train
- best balance of cost and time
- direct train only
- overnight recommendation
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

    # ------------------------------------------------------------------
    # Generic extraction helpers
    # ------------------------------------------------------------------

    def _get(self, obj: Any, *keys: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            for key in keys:
                if key in obj and obj[key] is not None:
                    return obj[key]
            return default
        for key in keys:
            if hasattr(obj, key):
                value = getattr(obj, key)
                if value is not None:
                    return value
        return default

    def _minutes(self, value: Any) -> float:
        if value is None:
            return 10**9
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        if text.isdigit():
            return float(text)
        import re
        total = 0.0
        h = re.search(r"(\d+)\s*h", text)
        m = re.search(r"(\d+)\s*m", text)
        if h:
            total += float(h.group(1)) * 60
        if m:
            total += float(m.group(1))
        if total > 0:
            return total
        if ":" in text:
            try:
                hh, mm = text.split(":", 1)
                return float(int(hh) * 60 + int(mm[:2]))
            except Exception:
                pass
        try:
            return float(text)
        except Exception:
            return 10**9

    def _int_value(self, value: Any, default: int = 10**9) -> int:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        try:
            return int(float(text))
        except Exception:
            return default

    def _is_direct(self, train: Any) -> bool:
        direct = self._get(train, "is_direct", "direct", "non_stop")
        if isinstance(direct, bool):
            return direct
        return False

    def _duration(self, train: Any) -> float:
        return self._minutes(self._get(train, "duration_minutes", "duration", "travel_time", "journey_time"))

    def _stops(self, train: Any) -> float:
        return float(self._int_value(self._get(train, "total_stops", "stops", "stop_count")))

    def _train_number(self, train: Any) -> str:
        return str(self._get(train, "train_number", default="") or "")

    def _train_name(self, train: Any) -> str:
        return str(self._get(train, "train_name", default="") or "")

    def _departure_hour(self, train: Any) -> Optional[int]:
        value = self._get(train, "departure_time", "departure", "dep", default=None)
        if not value:
            return None
        text = str(value).strip().lower()
        import re
        m = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", text)
        if not m:
            return None
        hh = int(m.group(1))
        ampm = m.group(3)
        if ampm == "pm" and hh != 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0
        return hh

    def _fare_estimate(
        self,
        train: Any,
        source_code: Optional[str],
        dest_code: Optional[str],
        travel_class: str = "SL",
        passengers: int = 1,
    ) -> Optional[float]:
        train_name = self._train_name(train)
        distance = self._get(train, "distance_km", "route_distance", default=None)
        try:
            breakdown = self.fare_calc.calculate(
                travel_class=travel_class,
                distance_km=int(distance) if distance is not None else None,
                train_name=train_name,
                passengers=passengers,
                source_code=source_code,
                dest_code=dest_code,
            )
            return float(breakdown.total_fare)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def score_trains(
        self,
        trains: Sequence[Any],
        source_code: Optional[str] = None,
        dest_code: Optional[str] = None,
        preference: Optional[str] = None,
        travel_class: str = "SL",
        passengers: int = 1,
    ) -> List[Recommendation]:
        """
        Score each train using a multi-factor heuristic.

        preference:
          - cheapest
          - fastest
          - direct
          - overnight
          - best_balance
        """
        prefs = (preference or "best_balance").lower().strip()
        recs: List[Recommendation] = []

        if not trains:
            return recs

        durations = [self._duration(t) for t in trains if self._duration(t) < 10**9]
        stops = [self._stops(t) for t in trains if self._stops(t) < 10**9]

        min_dur = min(durations) if durations else 1.0
        max_dur = max(durations) if durations else 1.0
        min_stops = min(stops) if stops else 0.0
        max_stops = max(stops) if stops else 1.0

        estimated_fares: List[Tuple[Any, Optional[float]]] = []
        for t in trains:
            estimated_fares.append(
                (t, self._fare_estimate(t, source_code, dest_code, travel_class, passengers))
            )

        fares = [f for _, f in estimated_fares if f is not None]
        min_fare = min(fares) if fares else 1.0
        max_fare = max(fares) if fares else 1.0

        for train, fare in estimated_fares:
            duration = self._duration(train)
            stop_count = self._stops(train)
            direct = self._is_direct(train)
            dep_hour = self._departure_hour(train)

            # Normalized scores in [0,1], higher is better.
            if fare is None:
                fare_score = 0.45
            else:
                fare_score = 1.0 - ((fare - min_fare) / max(1.0, (max_fare - min_fare)))

            duration_score = 1.0 - ((duration - min_dur) / max(1.0, (max_dur - min_dur))) if duration < 10**9 else 0.4
            stop_score = 1.0 - ((stop_count - min_stops) / max(1.0, (max_stops - min_stops))) if stop_count < 10**9 else 0.4
            direct_score = 1.0 if direct else max(0.0, 1.0 - (stop_count / max(1.0, max_stops + 1)))

            overnight_score = 0.5
            if dep_hour is not None:
                overnight_score = 1.0 if (20 <= dep_hour or dep_hour <= 6) else 0.2

            if prefs == "cheapest":
                score = fare_score * 0.72 + duration_score * 0.10 + direct_score * 0.10 + stop_score * 0.08
            elif prefs == "fastest":
                score = duration_score * 0.70 + fare_score * 0.15 + direct_score * 0.10 + stop_score * 0.05
            elif prefs == "direct":
                score = direct_score * 0.75 + duration_score * 0.15 + fare_score * 0.05 + stop_score * 0.05
            elif prefs == "overnight":
                score = overnight_score * 0.45 + duration_score * 0.25 + direct_score * 0.15 + fare_score * 0.15
            else:  # best_balance
                score = fare_score * 0.35 + duration_score * 0.35 + direct_score * 0.15 + stop_score * 0.15

            reasons: List[str] = []
            if direct:
                reasons.append("direct train")
            if fare is not None:
                reasons.append(f"est. fare ₹{fare:,.0f}")
            if duration < 10**9:
                reasons.append(f"{int(duration)} min")
            if stop_count < 10**9:
                reasons.append(f"{int(stop_count)} stops")
            if dep_hour is not None and (20 <= dep_hour or dep_hour <= 6):
                reasons.append("overnight-friendly")

            recs.append(
                Recommendation(
                    train=train,
                    score=round(score, 4),
                    reasons=reasons,
                    fare_estimate=fare,
                    duration_minutes=duration if duration < 10**9 else None,
                    stops=stop_count if stop_count < 10**9 else None,
                )
            )

        recs.sort(
            key=lambda r: (
                -r.score,
                r.duration_minutes if r.duration_minutes is not None else 10**9,
                r.stops if r.stops is not None else 10**9,
                self._train_number(r.train),
            )
        )
        return recs

    def recommend_best(
        self,
        trains: Sequence[Any],
        source_code: Optional[str] = None,
        dest_code: Optional[str] = None,
        preference: Optional[str] = None,
        travel_class: str = "SL",
        passengers: int = 1,
    ) -> Optional[Recommendation]:
        ranked = self.score_trains(
            trains,
            source_code=source_code,
            dest_code=dest_code,
            preference=preference,
            travel_class=travel_class,
            passengers=passengers,
        )
        return ranked[0] if ranked else None

    def compare_trains(
        self,
        trains: Sequence[Any],
        source_code: Optional[str] = None,
        dest_code: Optional[str] = None,
        travel_class: str = "SL",
        passengers: int = 1,
    ) -> List[Dict[str, Any]]:
        ranked = self.score_trains(
            trains,
            source_code=source_code,
            dest_code=dest_code,
            preference="best_balance",
            travel_class=travel_class,
            passengers=passengers,
        )
        output: List[Dict[str, Any]] = []
        for rec in ranked:
            t = rec.train
            output.append(
                {
                    "train_number": self._train_number(t),
                    "train_name": self._train_name(t),
                    "score": rec.score,
                    "fare_estimate": rec.fare_estimate,
                    "duration_minutes": rec.duration_minutes,
                    "stops": rec.stops,
                    "reasons": rec.reasons,
                    "is_direct": self._is_direct(t),
                }
            )
        return output

    def explain_recommendation(self, rec: Recommendation) -> str:
        train_number = self._train_number(rec.train)
        train_name = self._train_name(rec.train)
        reasons = ", ".join(rec.reasons[:4]) if rec.reasons else "balanced option"
        fare_text = f"₹{rec.fare_estimate:,.0f}" if rec.fare_estimate is not None else "fare unavailable"
        duration_text = f"{int(rec.duration_minutes)} min" if rec.duration_minutes is not None else "duration unavailable"
        return (
            f"**{train_number}** ({train_name}) — score {rec.score:.2f}. "
            f"{fare_text}, {duration_text}. Why: {reasons}."
        )

    def shortlist(
        self,
        trains: Sequence[Any],
        source_code: Optional[str] = None,
        dest_code: Optional[str] = None,
        preference: Optional[str] = None,
        travel_class: str = "SL",
        passengers: int = 1,
        top_n: int = 3,
    ) -> List[Recommendation]:
        ranked = self.score_trains(
            trains,
            source_code=source_code,
            dest_code=dest_code,
            preference=preference,
            travel_class=travel_class,
            passengers=passengers,
        )
        return ranked[: max(1, int(top_n))]
