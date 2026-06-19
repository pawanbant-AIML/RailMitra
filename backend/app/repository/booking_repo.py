"""booking_repo.py — Data access layer for bookings.

Fixes applied:
- Replaced deprecated payload.dict() with payload.model_dump() (Pydantic v2)
- Wrapped db.commit() in try/except with db.rollback() on failure
- list_by_user now orders by created_at descending (newest first)
"""
from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.train_models import Booking
from app.models import schemas

logger = logging.getLogger(__name__)


class BookingRepository:
    def create(self, payload: schemas.BookingCreate, db: Session) -> Booking:
        """Persist a new booking; raises ValueError on DB failure."""
        try:
            db_booking = Booking(**payload.model_dump())
            db.add(db_booking)
            db.commit()
            db.refresh(db_booking)
            logger.info("[booking_repo] Created booking id=%s train=%s", db_booking.id, db_booking.train_number)
            return db_booking
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("[booking_repo] DB error on create: %s", exc)
            raise ValueError(f"Could not save booking: {exc}") from exc

    def list_by_user(self, user_id: int, db: Session) -> List[Booking]:
        """Return all bookings for user, newest first."""
        return (
            db.query(Booking)
            .filter(Booking.user_id == user_id)
            .order_by(Booking.created_at.desc())
            .all()
        )

    def get_by_id(self, booking_id: int, db: Session) -> Optional[Booking]:
        return db.query(Booking).filter(Booking.id == booking_id).first()

    def cancel(self, booking_id: int, db: Session) -> bool:
        """Mark a booking as CANCELLED. Returns True on success, False if not found."""
        try:
            booking = db.query(Booking).filter(Booking.id == booking_id).first()
            if not booking:
                return False
            if booking.status == "CANCELLED":
                # Already cancelled — treat as success to be idempotent
                return True
            booking.status = "CANCELLED"
            db.commit()
            logger.info("[booking_repo] Cancelled booking id=%s", booking_id)
            return True
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("[booking_repo] DB error on cancel id=%s: %s", booking_id, exc)
            return False