from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.train_models import Route, Train


class TrainRepository:
    """Train lookup helpers for search, details, and route-aware queries."""

    def get_all(self, limit: int, db: Session):
        return db.query(Train).limit(max(1, limit)).all()

    def get_by_number(self, number: str, db: Session):
        if not number:
            return None
        return db.query(Train).filter(func.upper(Train.train_number) == number.strip().upper()).first()

    def search_by_number(self, query: str, db: Session, limit: int = 20):
        q = (query or "").strip()
        if not q:
            return []
        pattern = f"%{q}%"
        return (
            db.query(Train)
            .filter(func.upper(Train.train_number).like(func.upper(pattern)))
            .limit(max(1, limit))
            .all()
        )

    def search_by_name(self, query: str, db: Session, limit: int = 20):
        q = (query or "").strip().lower()
        if not q:
            return []
        return (
            db.query(Train)
            .filter(func.lower(Train.train_name).like(f"%{q}%"))
            .limit(max(1, limit))
            .all()
        )

    def search(self, query: str, db: Session, limit: int = 20):
        """Broad search by train number or name."""
        q = (query or "").strip()
        if not q:
            return []
        pattern = f"%{q.lower()}%"
        return (
            db.query(Train)
            .filter(
                or_(
                    func.lower(Train.train_name).like(pattern),
                    func.lower(Train.train_number).like(pattern),
                )
            )
            .limit(max(1, limit))
            .all()
        )

    def get_direct_trains(self, source_codes: Sequence[str], dest_codes: Sequence[str], db: Session, limit: int = 100):
        """Return trains where a source station occurs before a destination station."""
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

    def get_trains_serving_station(self, station_codes: Sequence[str], db: Session, limit: int = 100):
        """Return trains that stop at any of the provided station codes."""
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

    def get_trains_between(
        self,
        source_codes: Sequence[str],
        dest_codes: Sequence[str],
        db: Session,
        limit: int = 100,
    ):
        """
        Return trains that can travel from any source code to any destination code
        in the correct route order.
        """
        return self.get_direct_trains(source_codes, dest_codes, db, limit=limit)

    def get_by_station_pair(
        self,
        source_code: str,
        destination_code: str,
        db: Session,
        limit: int = 100,
    ):
        """Convenience wrapper for a single source/destination pair."""
        return self.get_direct_trains([source_code], [destination_code], db, limit=limit)

    def get_available_trains(self, db: Session, limit: int = 50):
        """
        Alias for a common UI use case: show any trains when the user asks
        "show all available options".
        """
        return self.get_all(limit, db)


__all__ = ["TrainRepository"]
