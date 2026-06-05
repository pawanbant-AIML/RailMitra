from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.models import schemas
from app.repository.booking_repo import BookingRepository
from app.api.v1.dependencies import get_db

router = APIRouter()
repo = BookingRepository()

@router.post("/bookings", response_model=schemas.Booking, status_code=status.HTTP_201_CREATED)
def create_booking(booking: schemas.BookingCreate, db: Session = Depends(get_db)):
    return repo.create(booking, db)

@router.get("/bookings", response_model=List[schemas.Booking])
def list_user_bookings(user_id: int, db: Session = Depends(get_db)):
    return repo.list_by_user(user_id, db)

@router.delete("/bookings/{booking_id}", response_model=dict)
def cancel_booking(booking_id: int, db: Session = Depends(get_db)):
    cancelled = repo.cancel(booking_id, db)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Booking not found")
    return {"status": "cancelled", "id": booking_id}