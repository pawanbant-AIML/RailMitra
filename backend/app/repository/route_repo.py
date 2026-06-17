from sqlalchemy.orm import Session
from typing import Optional
from app.models.train_models import Route


class RouteRepository:
    def get_by_train(self, train_number: str, db: Session):
        return (
            db.query(Route)
            .filter(Route.train_number == train_number)
            .order_by(Route.sequence)
            .all()
        )

    def get_distance_between(
        self,
        train_number: str,
        source_code: str,
        dest_code: str,
        db: Session,
    ) -> Optional[int]:
        """
        Calculate the distance (km) between two stations for a given train
        by subtracting their cumulative distance_km values from the routes table.
        Returns None if either station is not found on the route.
        """
        stops = self.get_by_train(train_number, db)
        src_dist = None
        dst_dist = None
        for stop in stops:
            if stop.station_code.upper() == source_code.upper() and stop.distance_km is not None:
                src_dist = stop.distance_km
            if stop.station_code.upper() == dest_code.upper() and stop.distance_km is not None:
                dst_dist = stop.distance_km
        if src_dist is not None and dst_dist is not None and dst_dist > src_dist:
            return dst_dist - src_dist
        return None

    def get_all_station_codes(self, train_number: str, db: Session) -> list:
        """Return a list of station codes (in sequence) for the given train."""
        stops = self.get_by_train(train_number, db)
        return [s.station_code for s in stops]