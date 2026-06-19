"""
api/v1/endpoints/bookings.py

Fixes applied:
- BookingConfirmationRequest now accepts both `train_number` and `train_selection` 
  (field alias so frontend can send either)
- Train matching: exact number → exact name → fuzzy/partial → any found train
- Proper error handling wrapping ValueError from booking_repo
- DELETE endpoint returns 200 (not silent 200 on not-found)
- Added GET /bookings/{id} for per-booking lookup
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
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
    travel_date: str = Field(..., description="ISO date string, e.g. 2026-06-20")
    travel_class: str = Field(..., description="Travel class, e.g. SL, 3A")
    passenger_count: int = Field(..., ge=1, le=12, description="Number of passengers (1–12)")
    # Accept both field names from frontend
    train_number: Optional[str] = Field(None, description="Train number")
    train_selection: Optional[str] = Field(None, description="Train number or name (alias)")
    user_id: int = Field(1, ge=1, description="User identifier")

    @model_validator(mode="after")
    def resolve_train_selection(self) -> "BookingConfirmationRequest":
        """Normalise: whichever of train_number / train_selection is set becomes train_selection."""
        if not self.train_selection and self.train_number:
            self.train_selection = self.train_number
        if not self.train_number and self.train_selection:
            self.train_number = self.train_selection
        return self


class BookingConfirmationResponse(BaseModel):
    success: bool
    status: str
    message: str
    booking: Optional[schemas.Booking] = None
    selected_train: Optional[schemas.Train] = None
    missing_fields: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------

@router.post("/bookings", response_model=schemas.Booking, status_code=status.HTTP_201_CREATED)
def create_booking(booking: schemas.BookingCreate, db: Session = Depends(get_db)):
    try:
        return repo.create(booking, db)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/bookings", response_model=List[schemas.Booking])
def list_user_bookings(user_id: int, db: Session = Depends(get_db)):
    return repo.list_by_user(user_id, db)


@router.get("/bookings/{booking_id}", response_model=schemas.Booking)
def get_booking(booking_id: int, db: Session = Depends(get_db)):
    booking = repo.get_by_id(booking_id, db)
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking


@router.delete("/bookings/{booking_id}", response_model=dict)
def cancel_booking(booking_id: int, db: Session = Depends(get_db)):
    cancelled = repo.cancel(booking_id, db)
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"Booking #{booking_id} not found")
    return {"status": "cancelled", "id": booking_id}


# ---------------------------------------------------------------------------
# Confirm endpoint (called from BookingDrawer form submit)
# ---------------------------------------------------------------------------

@router.post(
    "/bookings/confirm",
    response_model=BookingConfirmationResponse,
    status_code=status.HTTP_201_CREATED,
)
def confirm_booking(payload: BookingConfirmationRequest, db: Session = Depends(get_db)):
    logger.info(
        "[bookings/confirm] source=%s dest=%s date=%s class=%s pax=%d train=%s user=%d",
        payload.source,
        payload.destination,
        payload.travel_date,
        payload.travel_class,
        payload.passenger_count,
        payload.train_selection,
        payload.user_id,
    )

    # ---- Field validation ----
    missing_fields: List[str] = []
    errors: List[str] = []

    source = (payload.source or "").strip()
    destination = (payload.destination or "").strip()
    travel_date = (payload.travel_date or "").strip()
    travel_class = (payload.travel_class or "").strip()
    train_selection = (payload.train_selection or "").strip()

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
        logger.warning("[bookings/confirm] Missing fields: %s", missing_fields)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "status": "failed",
                "message": f"Missing required fields: {', '.join(missing_fields)}.",
                "missing_fields": missing_fields,
                "errors": errors or ["Please fill all required fields."],
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

        # --- Multi-strategy train matching ---
        selected_train = None
        train_sel_upper = train_selection.upper()

        # 1. Exact train number match
        for t in trains:
            if str(getattr(t, "train_number", "")).strip().upper() == train_sel_upper:
                selected_train = t
                break

        # 2. Exact name match
        if not selected_train:
            for t in trains:
                if str(getattr(t, "train_name", "")).strip().upper() == train_sel_upper:
                    selected_train = t
                    break

        # 3. Partial / contains match (train_number substring)
        if not selected_train:
            for t in trains:
                num = str(getattr(t, "train_number", "")).strip().upper()
                name = str(getattr(t, "train_name", "")).strip().upper()
                if train_sel_upper in num or train_sel_upper in name:
                    selected_train = t
                    break

        # 4. Absolute fallback — train not found matching the criteria
        if not selected_train:
            logger.warning(
                "[bookings/confirm] No valid trains found matching selection %s for route %s→%s",
                train_selection, src_code, dst_code
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "status": "failed",
                    "message": (
                        f"No trains found from {source} to {destination}. "
                        "Please verify the station names and try again."
                    ),
                    "missing_fields": [],
                    "errors": ["No matching trains for this route."],
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
            "[bookings/confirm] Confirmed booking_id=%s train=%s",
            getattr(booking, "id", None),
            booking_payload.train_number,
        )

        return BookingConfirmationResponse(
            success=True,
            status="confirmed",
            message="Booking confirmed successfully.",
            booking=booking,
            selected_train=selected_train,
        )

    except ValueError as exc:
        logger.error("[bookings/confirm] ValueError: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "status": "failed",
                "message": str(exc),
                "missing_fields": [],
                "errors": [str(exc)],
                "booking": None,
                "selected_train": None,
            },
        )
    except Exception as exc:
        logger.exception("[bookings/confirm] Unexpected error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "status": "failed",
                "message": "Unable to confirm booking. Please try again.",
                "missing_fields": [],
                "errors": [f"{type(exc).__name__}: {exc}"],
                "booking": None,
                "selected_train": None,
            },
        )
