from sqlalchemy.orm import Session
from app.models.train_models import Booking
from app.models import schemas


class BookingRepository:
    def create(self, payload: schemas.BookingCreate, db: Session):
        db_booking = Booking(**payload.dict())
        db.add(db_booking)
        db.commit()
        db.refresh(db_booking)
        return db_booking

    def list_by_user(self, user_id: int, db: Session):
        return db.query(Booking).filter(Booking.user_id == user_id).all()

    def cancel(self, booking_id: int, db: Session):
        booking = db.query(Booking).filter(Booking.id == booking_id).first()
        if not booking:
            return False
        booking.status = "CANCELLED"
        db.commit()
        return True