from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from app.models.train_models import Route, Train


class RouteRepository:
    """Route lookup helpers for direct, through, and ranked route access."""

    def get_by_train(self, train_number: str, db: Session):
        return (
            db.query(Route)
            .filter(Route.train_number == train_number)
            .order_by(Route.sequence)
            .all()
        )

    def get_stop_count(self, train_number: str, db: Session) -> int:
        """Return the number of route stops for a train."""
        return len(self.get_by_train(train_number, db))

    def get_all_station_codes(self, train_number: str, db: Session) -> list[str]:
        """Return station codes in route sequence for a train."""
        stops = self.get_by_train(train_number, db)
        return [s.station_code for s in stops if s.station_code]

    def get_station_order(self, train_number: str, station_code: str, db: Session) -> Optional[int]:
        """Return the 1-based sequence order for a station on a given train."""
        for stop in self.get_by_train(train_number, db):
            if stop.station_code and stop.station_code.upper() == station_code.upper():
                return int(stop.sequence or 0) or None
        return None

    def get_distance_between(
        self,
        train_number: str,
        source_code: str,
        dest_code: str,
        db: Session,
    ) -> Optional[int]:
        """
        Calculate distance between two stations for a train by subtracting
        cumulative distances from the route table.
        """
        stops = self.get_by_train(train_number, db)
        src_dist = None
        dst_dist = None
        for stop in stops:
            if stop.station_code and stop.station_code.upper() == source_code.upper() and stop.distance_km is not None:
                src_dist = stop.distance_km
            if stop.station_code and stop.station_code.upper() == dest_code.upper() and stop.distance_km is not None:
                dst_dist = stop.distance_km
        if src_dist is not None and dst_dist is not None and dst_dist > src_dist:
            return int(dst_dist - src_dist)
        return None

    def get_stations_between(
        self,
        train_number: str,
        source_code: str,
        dest_code: str,
        db: Session,
    ) -> List[Route]:
        """Return ordered route stops between source and destination inclusive."""
        stops = self.get_by_train(train_number, db)
        src_seq = None
        dst_seq = None
        for stop in stops:
            code = (stop.station_code or "").upper()
            if code == source_code.upper():
                src_seq = stop.sequence
            if code == dest_code.upper():
                dst_seq = stop.sequence
        if src_seq is None or dst_seq is None or dst_seq <= src_seq:
            return []
        return [s for s in stops if src_seq <= s.sequence <= dst_seq]

    def has_direct_connection(
        self,
        train_number: str,
        source_code: str,
        dest_code: str,
        db: Session,
    ) -> bool:
        """True when a train visits both stations in the correct order."""
        return self.get_distance_between(train_number, source_code, dest_code, db) is not None

    def trains_passing_through(
        self,
        station_codes: Sequence[str],
        db: Session,
        limit: int = 50,
    ) -> List[Train]:
        """Return trains that stop at any of the station codes."""
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

    def find_direct_trains(
        self,
        source_codes: Sequence[str],
        dest_codes: Sequence[str],
        db: Session,
        limit: int = 100,
    ) -> List[Train]:
        """
        Return trains where a source station appears before a destination station.
        Useful for direct train search and recommendation ranking.
        """
        src = [c.upper() for c in source_codes if c]
        dst = [c.upper() for c in dest_codes if c]
        if not src or not dst:
            return []

        RouteSrc = Route
        RouteDst = Route

        subq_src = (
            db.query(RouteSrc.train_number.label("train_number"), RouteSrc.sequence.label("sequence"))
            .filter(RouteSrc.station_code.in_(src))
            .subquery()
        )
        subq_dst = (
            db.query(RouteDst.train_number.label("train_number"), RouteDst.sequence.label("sequence"))
            .filter(RouteDst.station_code.in_(dst))
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

    def get_candidate_route_pairs(
        self,
        source_codes: Sequence[str],
        dest_codes: Sequence[str],
        db: Session,
        limit: int = 100,
    ) -> List[Tuple[str, str, str]]:
        """
        Return (train_number, source_code, destination_code) triples for route discovery.
        This is useful when you need to compare multiple stations/cities.
        """
        src = [c.upper() for c in source_codes if c]
        dst = [c.upper() for c in dest_codes if c]
        if not src or not dst:
            return []

        rows = (
            db.query(Route.train_number, Route.station_code, Route.sequence)
            .filter(Route.station_code.in_(list(set(src + dst))))
            .order_by(Route.train_number, Route.sequence)
            .all()
        )

        grouped: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for train_number, station_code, sequence in rows:
            grouped[str(train_number)].append((str(station_code), int(sequence or 0)))

        triples: List[Tuple[str, str, str]] = []
        src_set = set(src)
        dst_set = set(dst)
        for train_number, stops in grouped.items():
            stations = [s for s, _ in stops]
            seq_map = {s: seq for s, seq in stops}
            for s in src_set:
                for d in dst_set:
                    if s in seq_map and d in seq_map and seq_map[d] > seq_map[s]:
                        triples.append((train_number, s, d))
                        if len(triples) >= limit:
                            return triples
        return triples

    def route_summary(self, train_number: str, db: Session) -> dict:
        """Return a small structured summary for downstream ranking or display."""
        stops = self.get_by_train(train_number, db)
        if not stops:
            return {
                "status": "no_results",
                "train_number": train_number,
                "stops": [],
            }

        first = stops[0]
        last = stops[-1]
        return {
            "status": "ok",
            "train_number": train_number,
            "stop_count": len(stops),
            "start_station": first.station_code,
            "end_station": last.station_code,
            "start_distance_km": first.distance_km,
            "end_distance_km": last.distance_km,
            "stops": [
                {
                    "seq": s.sequence,
                    "station_code": s.station_code,
                    "arrival": s.arrival_time or "--:--",
                    "departure": s.departure_time or "--:--",
                    "distance_km": s.distance_km,
                }
                for s in stops
            ],
        }


__all__ = ["RouteRepository"]
