from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from app.models import schemas
from app.repository.train_repo import TrainRepository
from app.api.v1.dependencies import get_db

router = APIRouter()
repo   = TrainRepository()


@router.get("/trains", response_model=List[schemas.Train])
def list_trains(limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    return repo.get_all(limit, db)


@router.get("/trains/search", response_model=List[schemas.Train])
def search_trains_by_route(
    from_station: str = Query(..., description="Source station name or code, e.g. Bangalore or SBC"),
    to_station:   str = Query(..., description="Destination station name or code, e.g. Mumbai or CSMT"),
    db: Session = Depends(get_db),
):
    """
    Find all trains that run from *from_station* to *to_station*.
    Accepts city names (Bangalore, Mumbai …) **or** station codes (SBC, CSMT …).
    """
    from app.services.timetable_service import TimetableService
    from app.repository.station_repo import StationRepository

    station_repo = StationRepository()
    timetable    = TimetableService()

    src_code = station_repo.fuzzy_find_station(from_station, db) or from_station.upper()
    dst_code = station_repo.fuzzy_find_station(to_station,   db) or to_station.upper()

    results = timetable.search(src_code, dst_code, None, db)
    return results


@router.get("/trains/{train_number}", response_model=schemas.Train)
def get_train(train_number: str, db: Session = Depends(get_db)):
    train = repo.get_by_number(train_number, db)
    if not train:
        raise HTTPException(status_code=404, detail="Train not found")
    return train