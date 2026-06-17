from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import and_, or_
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
    """
    Search and ranking layer over train, route, and station repositories.

    This version preserves the existing `search(src, dst, date, db)` API and adds
    ranking helpers for fastest/cheapest/direct options as well as route discovery.
    """

    def __init__(self) -> None:
        self.train_repo = TrainRepository()
        self.station_repo = StationRepository()
        self.route_repo = RouteRepository()

    # ------------------------------------------------------------------
    # Station resolution
    # ------------------------------------------------------------------

    def _expand_codes(self, code_or_city: str, db: Session) -> List[str]:
        codes: List[str] = []
        if not code_or_city:
            return codes

        raw = str(code_or_city).strip()
        if not raw:
            return codes

        try:
            city_codes = self.station_repo.get_all_codes_for_city(raw, db) or []
            for c in city_codes:
                if c and str(c).upper() not in codes:
                    codes.append(str(c).upper())
        except Exception:
            pass

        upper = raw.upper()
        if upper not in codes:
            codes.append(upper)

        try:
            resolved = self.station_repo.fuzzy_find_station(raw, db)
            if resolved and str(resolved).upper() not in codes:
                codes.append(str(resolved).upper())
        except Exception:
            pass

        return codes

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(self, src: str, dst: str, date: Optional[str], db: Session) -> List[Train]:
        """
        Find trains that connect src → dst (station codes or names).

        Search strategy:
          1. Expand source/destination to all matching station codes in the same city
          2. Direct train table match
          3. Route-based through-train search
          4. Train repository fallback if available
        """
        src_codes = self._expand_codes(src, db)
        dst_codes = self._expand_codes(dst, db)

        if not src_codes or not dst_codes:
            return []

        results: List[Train] = []
        seen: set = set()

        # Direct table match
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

        # Route-based through-running trains
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

        # Repository fallback
        if not results:
            for method_name in ("search_between", "search_by_route", "find_between"):
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

        return results

    def search_direct(self, src: str, dst: str, date: Optional[str], db: Session) -> List[Train]:
        src_codes = self._expand_codes(src, db)
        dst_codes = self._expand_codes(dst, db)
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
        return results

    def search_through_trains(self, src: str, dst: str, db: Session) -> List[Train]:
        src_codes = self._expand_codes(src, db)
        dst_codes = self._expand_codes(dst, db)
        results: List[Train] = []
        seen: set = set()
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
        return results

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------

    def _safe_float(self, value: Any, default: float = 10**9) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower()
        if text.isdigit():
            return float(text)
        minutes = 0.0
        import re
        h = re.search(r"(\d+)\s*h", text)
        m = re.search(r"(\d+)\s*m", text)
        if h:
            minutes += float(h.group(1)) * 60
        if m:
            minutes += float(m.group(1))
        if minutes > 0:
            return minutes
        try:
            return float(text)
        except Exception:
            return default

    def _train_duration_minutes(self, train: Any) -> float:
        for key in ("duration_minutes", "duration", "travel_time", "journey_time"):
            value = getattr(train, key, None)
            if value is not None:
                return self._safe_float(value)
        return 10**9

    def _train_stops(self, train: Any) -> float:
        for key in ("total_stops", "stops", "stop_count"):
            value = getattr(train, key, None)
            if value is not None:
                return self._safe_float(value)
        return 10**9

    def _train_number(self, train: Any) -> str:
        return str(getattr(train, "train_number", "") or "")

    def _train_name(self, train: Any) -> str:
        return str(getattr(train, "train_name", "") or "")

    def _is_direct(self, train: Any) -> bool:
        for key in ("is_direct", "direct", "non_stop"):
            value = getattr(train, key, None)
            if isinstance(value, bool) and value:
                return True
        return False

    def rank_by_fastest(self, trains: Sequence[Any]) -> List[Any]:
        return sorted(
            list(trains),
            key=lambda t: (
                self._train_duration_minutes(t),
                self._train_stops(t),
                self._train_number(t),
            ),
        )

    def rank_by_fewest_stops(self, trains: Sequence[Any]) -> List[Any]:
        return sorted(
            list(trains),
            key=lambda t: (
                self._train_stops(t),
                self._train_duration_minutes(t),
                self._train_number(t),
            ),
        )

    def rank_by_directness(self, trains: Sequence[Any]) -> List[Any]:
        return sorted(
            list(trains),
            key=lambda t: (
                0 if self._is_direct(t) else 1,
                self._train_stops(t),
                self._train_duration_minutes(t),
                self._train_number(t),
            ),
        )

    def search_with_preferences(
        self,
        src: str,
        dst: str,
        date: Optional[str],
        db: Session,
        sort_by: Optional[str] = None,
        limit: Optional[int] = None,
        require_direct: bool = False,
    ) -> List[Train]:
        trains = self.search(src, dst, date, db)
        if require_direct:
            trains = [t for t in trains if self._is_direct(t) or self._train_stops(t) <= 2]

        if sort_by == "duration":
            trains = self.rank_by_fastest(trains)
        elif sort_by == "stops":
            trains = self.rank_by_fewest_stops(trains)
        elif sort_by == "direct":
            trains = self.rank_by_directness(trains)

        if limit:
            trains = trains[: max(1, int(limit))]
        return trains

    def get_route_gap(self, train_number: str, src: str, dst: str, db: Session) -> Optional[Tuple[int, int]]:
        """
        Return (source_seq, dest_seq) if the train passes through both stations.
        """
        src_codes = self._expand_codes(src, db)
        dst_codes = self._expand_codes(dst, db)
        try:
            rows = (
                db.query(RouteModel.sequence, RouteModel.station_code)
                .filter(RouteModel.train_number == str(train_number))
                .filter(or_(RouteModel.station_code.in_(src_codes), RouteModel.station_code.in_(dst_codes)))
                .all()
            )
            source_seq = None
            dest_seq = None
            for seq, station in rows:
                if station in src_codes and source_seq is None:
                    source_seq = int(seq)
                if station in dst_codes and dest_seq is None:
                    dest_seq = int(seq)
            if source_seq is not None and dest_seq is not None:
                return source_seq, dest_seq
        except Exception:
            pass
        return None

    def get_train_summary(self, train: Any) -> Dict[str, Any]:
        return {
            "train_number": self._train_number(train),
            "train_name": self._train_name(train),
            "source_station_code": getattr(train, "source_station_code", None),
            "destination_station_code": getattr(train, "destination_station_code", None),
            "departure": getattr(train, "departure_time", None) or getattr(train, "departure", None),
            "arrival": getattr(train, "arrival_time", None) or getattr(train, "arrival", None),
            "duration": getattr(train, "duration", None) or getattr(train, "travel_time", None),
            "stops": getattr(train, "total_stops", None) or getattr(train, "stops", None),
            "is_direct": self._is_direct(train),
        }
