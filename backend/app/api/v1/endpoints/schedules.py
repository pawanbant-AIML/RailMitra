from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.models import schemas
from app.repository.route_repo import RouteRepository
from app.api.v1.dependencies import get_db

router = APIRouter()
repo = RouteRepository()

@router.get("/schedules/{train_number}", response_model=List[schemas.Route])
def get_schedule(train_number: str, db: Session = Depends(get_db)):
    schedule = repo.get_by_train(train_number, db)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule