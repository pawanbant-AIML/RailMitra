from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional
from app.repository.train_repo import TrainRepository
from app.repository.station_repo import StationRepository
from app.repository.route_repo import RouteRepository
from app.models.train_models import Train


class TimetableService:
    def __init__(self):
        self.train_repo   = TrainRepository()
        self.station_repo = StationRepository()
        self.route_repo   = RouteRepository()

    def search(self, src: str, dst: str, date: Optional[str], db: Session) -> List[Train]:
        """
        Find trains that connect src → dst (station codes).
        Strategy:
          1. Expand src/dst to ALL station codes for that city.
          2. Direct match on trains.source_station_code / destination_station_code
          3. Route-based search for through-trains
        Returns ORM Train objects.
        """
        # Get all codes for the city (e.g. "SBC" → ["SBC","YPR","BNC","BNCE"])
        src_codes = self.station_repo.get_all_codes_for_city(src, db)
        dst_codes = self.station_repo.get_all_codes_for_city(dst, db)

        # Also include the code itself
        if src.upper() not in [c.upper() for c in src_codes]:
            src_codes.append(src.upper())
        if dst.upper() not in [c.upper() for c in dst_codes]:
            dst_codes.append(dst.upper())

        results: List[Train] = []
        seen: set = set()

        # ── Strategy 1: Direct trains table match ──────────────────────
        direct = (
            db.query(Train)
            .filter(
                Train.source_station_code.in_(src_codes),
                Train.destination_station_code.in_(dst_codes),
            )
            .all()
        )
        for t in direct:
            if t.train_number not in seen:
                results.append(t)
                seen.add(t.train_number)

        # ── Strategy 2: Route-based search (through-running trains) ────
        # Find trains that stop at any src code BEFORE any dst code
        from app.models.train_models import Route as RouteModel
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

        from sqlalchemy import and_
        route_trains = (
            db.query(Train)
            .join(subq_src, Train.train_number == subq_src.c.train_number)
            .join(subq_dst, Train.train_number == subq_dst.c.train_number)
            .filter(subq_src.c.sequence < subq_dst.c.sequence)
            .all()
        )
        for t in route_trains:
            if t.train_number not in seen:
                results.append(t)
                seen.add(t.train_number)

        return results