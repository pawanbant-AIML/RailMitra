from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from app.repository.station_repo import StationRepository
from app.models.train_models import Train   # <-- import Train model directly
from app.api.v1.dependencies import get_db

router = APIRouter()

@router.get("/search")
def global_search(
    q: str = Query(..., description="Search query (station name, code, or train number/name)"),
    type: Optional[str] = Query(None, description="Filter: 'stations' or 'trains'"),
    db: Session = Depends(get_db),
):
    """
    Unified search endpoint – returns matching stations and/or trains.
    """
    results = {"stations": [], "trains": []}

    if type is None or type == "stations":
        repo = StationRepository()
        stations = repo.search(q, limit=10, db=db)
        results["stations"] = stations

    if type is None or type == "trains":
        # Query Train model directly – no repo.model needed
        trains = (
            db.query(Train)
            .filter(
                (Train.train_number.ilike(f"%{q}%")) |
                (Train.train_name.ilike(f"%{q}%"))
            )
            .limit(10)
            .all()
        )
        results["trains"] = trains

    return results
