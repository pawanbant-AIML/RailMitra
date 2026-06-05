from sqlalchemy.orm import Session
from app.models.train_models import Fare


class FareRepository:
    def get_by_train(self, train_number: str, db: Session):
        return db.query(Fare).filter(Fare.train_number == train_number).all()