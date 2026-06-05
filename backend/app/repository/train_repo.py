from sqlalchemy.orm import Session
from app.models.train_models import Train


class TrainRepository:
    def get_all(self, limit: int, db: Session):
        return db.query(Train).limit(limit).all()

    def get_by_number(self, number: str, db: Session):
        return db.query(Train).filter(Train.train_number == number).first()