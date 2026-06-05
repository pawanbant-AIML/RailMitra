from sqlalchemy.orm import Session
from app.models.train_models import Route


class RouteRepository:
    def get_by_train(self, train_number: str, db: Session):
        return (
            db.query(Route)
            .filter(Route.train_number == train_number)
            .order_by(Route.sequence)
            .all()
        )