from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from app.models import schemas
from app.repository.station_repo import StationRepository
from app.api.v1.dependencies import get_db

router = APIRouter()
repo = StationRepository()

@router.get("/stations", response_model=List[schemas.Station])
def search_stations(q: str = Query(..., description="Partial station name or code"),
                    limit: int = Query(20, ge=1, le=100),
                    db: Session = Depends(get_db)):
    return repo.search(q, limit, db)

@router.get("/stations/{code}", response_model=schemas.Station)
def get_station(code: str, db: Session = Depends(get_db)):
    station = repo.get_by_code(code, db)
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")
    return station