from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.models import schemas
from app.repository.fare_repo import FareRepository
from app.api.v1.dependencies import get_db

router = APIRouter()
repo = FareRepository()

@router.get("/fares/{train_number}", response_model=List[schemas.Fare])
def list_fares(train_number: str, db: Session = Depends(get_db)):
    fares = repo.get_by_train(train_number, db)
    if not fares:
        raise HTTPException(status_code=404, detail="No fare data")
    return fares