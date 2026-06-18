from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.v1.dependencies import get_db
from app.models import schemas
from app.repository.booking_repo import BookingRepository
from app.services.booking_service import BookingService

logger = logging.getLogger(__name__)

router = APIRouter()
repo = BookingRepository()
booking_service = BookingService()


class BookingConfirmationRequest(BaseModel):
    source: str = Field(..., description="Source station name or code")
    destination: str = Field(..., description="Destination station name or code")
    travel_date: str = Field(..., description="ISO date string or datetime string")
    travel_class: str = Field(..., description="Passenger travel class, e.g. SL")
    passenger_count: int = Field(..., ge=1, description="Number of passengers")
    train_selection: str = Field(..., description="Selected train number or train name")
    user_id: int = Field(1, ge=1, description="User identifier")


class BookingConfirmationResponse(BaseModel):
    success: bool
    status: str
    message: str
    booking: Optional[schemas.Booking] = None
    selected_train: Optional[schemas.Train] = None
    missing_fields: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


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


@router.post(
    "/bookings/confirm",
    response_model=BookingConfirmationResponse,
    status_code=status.HTTP_201_CREATED,
)
def confirm_booking(payload: BookingConfirmationRequest, db: Session = Depends(get_db)):
    logger.info(
        "Booking confirmation requested",
        extra={
            "source": payload.source,
            "destination": payload.destination,
            "travel_date": payload.travel_date,
            "travel_class": payload.travel_class,
            "passenger_count": payload.passenger_count,
            "train_selection": payload.train_selection,
            "user_id": payload.user_id,
        },
    )

    missing_fields: List[str] = []
    errors: List[str] = []

    source = payload.source.strip()
    destination = payload.destination.strip()
    travel_date = payload.travel_date.strip()
    travel_class = payload.travel_class.strip()
    train_selection = payload.train_selection.strip()

    if not source:
        missing_fields.append("source")
    if not destination:
        missing_fields.append("destination")
    if not travel_date:
        missing_fields.append("travel_date")
    if not travel_class:
        missing_fields.append("travel_class")
    if payload.passenger_count < 1:
        missing_fields.append("passenger_count")
        errors.append("Passenger count must be at least 1.")
    if not train_selection:
        missing_fields.append("train_selection")

    if missing_fields:
        message = "Missing required booking details."
        logger.warning("Booking confirmation rejected: %s", ", ".join(missing_fields))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "status": "failed",
                "message": message,
                "missing_fields": missing_fields,
                "errors": errors or [message],
                "booking": None,
                "selected_train": None,
            },
        )

    try:
        src_code = booking_service._resolve_station(source, db) or source.upper()
        dst_code = booking_service._resolve_station(destination, db) or destination.upper()

        trains = booking_service.search_trains(
            {
                "source_station": src_code,
                "destination_station": dst_code,
                "date": travel_date,
            },
            db,
        )

        selected_train = None
        for train in trains:
            train_number = str(getattr(train, "train_number", "")).strip()
            train_name = str(getattr(train, "train_name", "")).strip()
            if train_selection == train_number or train_selection == train_name:
                selected_train = train
                break

        if selected_train is None and train_selection:
            logger.warning(
                "Selected train was not found in route search results",
                extra={
                    "source": src_code,
                    "destination": dst_code,
                    "train_selection": train_selection,
                    "matches": len(trains),
                },
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "status": "failed",
                    "message": "Selected train is not available for the chosen route and date.",
                    "missing_fields": [],
                    "errors": ["Selected train is not available for the chosen route and date."],
                    "booking": None,
                    "selected_train": None,
                },
            )

        booking_payload = schemas.BookingCreate(
            user_id=payload.user_id,
            train_number=str(getattr(selected_train, "train_number", train_selection)).strip(),
            passenger_count=payload.passenger_count,
            travel_class=booking_service._normalize_class(travel_class),
            travel_date=booking_service._normalize_date(travel_date),
        )

        booking = repo.create(booking_payload, db)

        logger.info(
            "Booking confirmed",
            extra={
                "booking_id": getattr(booking, "id", None),
                "train_number": booking_payload.train_number,
            },
        )

        return BookingConfirmationResponse(
            success=True,
            status="confirmed",
            message="Booking confirmed successfully.",
            booking=booking,
            selected_train=selected_train,
        )

    except Exception as exc:
        logger.exception("Booking confirmation failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "status": "failed",
                "message": "Unable to confirm booking.",
                "missing_fields": [],
                "errors": [str(exc)],
                "booking": None,
                "selected_train": None,
            },
        )
