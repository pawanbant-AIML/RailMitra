from __future__ import annotations

from typing import Any, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.models.train_models import Route, Train


class RouteRepository:
    """Route lookup helpers for direct, through, and time-aware route access."""

    def get_by_train(self, train_number: str, db: Session):
        return (
            db.query(Route)
            .filter(Route.train_number == train_number)
            .order_by(Route.sequence)
            .all()
        )

    def get_stop_count(self, train_number: str, db: Session) -> int:
        return len(self.get_by_train(train_number, db))

    def get_all_station_codes(self, train_number: str, db: Session) -> list[str]:
        stops = self.get_by_train(train_number, db)
        return [s.station_code for s in stops if getattr(s, "station_code", None)]

    def get_station_order(self, train_number: str, station_code: str, db: Session) -> Optional[int]:
        for stop in self.get_by_train(train_number, db):
            if getattr(stop, "station_code", None) and stop.station_code.upper() == station_code.upper():
                return int(stop.sequence or 0) or None
        return None

    def _stop_for_station(self, train_number: str, station_code: str, db: Session) -> Optional[Route]:
        for stop in self.get_by_train(train_number, db):
            if getattr(stop, "station_code", None) and stop.station_code.upper() == station_code.upper():
                return stop
        return None

    def _normalize_time(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text in {"--", "NA", "N/A"}:
            return None
        if len(text) == 5 and text[2] == ":":
            return text
        if len(text) == 4 and text.isdigit():
            return f"{text[:2]}:{text[2:]}"
        return text

    def get_departure_time(self, train_number: str, station_code: str, db: Session) -> Optional[str]:
        stop = self._stop_for_station(train_number, station_code, db)
        if stop is None:
            return None
        return self._normalize_time(getattr(stop, "departure_time", None))

    def get_arrival_time(self, train_number: str, station_code: str, db: Session) -> Optional[str]:
        stop = self._stop_for_station(train_number, station_code, db)
        if stop is None:
            return None
        return self._normalize_time(getattr(stop, "arrival_time", None))

    def get_distance_between(self, train_number: str, source_code: str, dest_code: str, db: Session) -> Optional[int]:
        stops = self.get_by_train(train_number, db)
        src_dist = None
        dst_dist = None
        for stop in stops:
            code = getattr(stop, "station_code", None)
            if not code:
                continue
            if code.upper() == source_code.upper() and getattr(stop, "distance_km", None) is not None:
                src_dist = stop.distance_km
            if code.upper() == dest_code.upper() and getattr(stop, "distance_km", None) is not None:
                dst_dist = stop.distance_km
        if src_dist is not None and dst_dist is not None and dst_dist > src_dist:
            return int(dst_dist - src_dist)
        return None

    def get_stations_between(self, train_number: str, source_code: str, dest_code: str, db: Session) -> List[Route]:
        stops = self.get_by_train(train_number, db)
        src_seq = None
        dst_seq = None
        for stop in stops:
            code = getattr(stop, "station_code", None)
            if not code:
                continue
            if code.upper() == source_code.upper():
                src_seq = stop.sequence
            if code.upper() == dest_code.upper():
                dst_seq = stop.sequence
        if src_seq is None or dst_seq is None or dst_seq <= src_seq:
            return []
        return [s for s in stops if src_seq <= s.sequence <= dst_seq]

    def has_direct_connection(self, train_number: str, source_code: str, dest_code: str, db: Session) -> bool:
        return self.get_distance_between(train_number, source_code, dest_code, db) is not None

    def trains_passing_through(self, station_codes: Sequence[str], db: Session, limit: int = 50) -> List[Train]:
        codes = [c.upper() for c in station_codes if c]
        if not codes:
            return []
        return (
            db.query(Train)
            .join(Route, Train.train_number == Route.train_number)
            .filter(Route.station_code.in_(codes))
            .distinct()
            .limit(limit)
            .all()
        )

    def find_direct_trains(self, source_codes: Sequence[str], dest_codes: Sequence[str], db: Session, limit: int = 100) -> List[Train]:
        src = [c.upper() for c in source_codes if c]
        dst = [c.upper() for c in dest_codes if c]
        if not src or not dst:
            return []

        subq_src = (
            db.query(Route.train_number.label("train_number"), Route.sequence.label("sequence"))
            .filter(Route.station_code.in_(src))
            .subquery()
        )
        subq_dst = (
            db.query(Route.train_number.label("train_number"), Route.sequence.label("sequence"))
            .filter(Route.station_code.in_(dst))
            .subquery()
        )

        return (
            db.query(Train)
            .join(subq_src, Train.train_number == subq_src.c.train_number)
            .join(subq_dst, Train.train_number == subq_dst.c.train_number)
            .filter(subq_src.c.sequence < subq_dst.c.sequence)
            .distinct()
            .limit(limit)
            .all()
        )

    def get_route_summary(self, train_number: str, db: Session) -> List[dict]:
        stops = self.get_by_train(train_number, db)
        out = []
        for stop in stops:
            out.append({
                "seq": getattr(stop, "sequence", None),
                "station_code": getattr(stop, "station_code", None),
                "arrival": self._normalize_time(getattr(stop, "arrival_time", None)) or "--:--",
                "departure": self._normalize_time(getattr(stop, "departure_time", None)) or "--:--",
                "distance_km": getattr(stop, "distance_km", None),
            })
        return out
