"""
agent/tools.py – LangChain-compatible tool definitions for RailMitra.

Each tool wraps a DB-backed service function.  Tools receive and return plain
strings so the LLM can parse them easily. All database errors are caught and
returned as error strings (so the LLM can decide how to handle them) rather
than raising exceptions.

DB session injection strategy:
  Tools are class methods on AgentTools.  The class is instantiated once per
  request inside AgentService, receiving the SQLAlchemy Session at that point.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Optional

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.models.train_models import Booking as BookingModel
from app.models import schemas
from app.repository.booking_repo import BookingRepository
from app.repository.fare_repo import FareRepository
from app.repository.route_repo import RouteRepository
from app.repository.station_repo import StationRepository
from app.repository.train_repo import TrainRepository
from app.services.booking_service import BookingService
from app.services.fare_calculator import FareCalculator
from app.services.timetable_service import TimetableService
from app.core.logger import logger

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CLASS_NAMES = {
    "GN": "General",
    "2S": "Second Sitting",
    "SL": "Sleeper",
    "CC": "Chair Car",
    "3A": "AC 3-Tier",
    "2A": "AC 2-Tier",
    "1A": "AC First Class",
    "EC": "Executive Chair Car",
}


def _norm_class(cls: str) -> str:
    """Normalise a user-friendly class string to the internal code."""
    mapping = {
        "sleeper": "SL", "sl": "SL",
        "general": "GN", "gn": "GN", "unreserved": "GN",
        "3ac": "3A", "3a": "3A", "third ac": "3A", "3 tier": "3A",
        "2ac": "2A", "2a": "2A", "second ac": "2A", "2 tier": "2A",
        "1ac": "1A", "1a": "1A", "first ac": "1A", "first class": "1A",
        "cc": "CC", "chair car": "CC",
        "ec": "EC", "executive": "EC",
        "2s": "2S", "second sitting": "2S",
    }
    return mapping.get(cls.lower().strip(), cls.upper().strip())


# ---------------------------------------------------------------------------
# AgentTools – bound to one DB session per request
# ---------------------------------------------------------------------------

class AgentTools:
    """
    Instantiate once per chat request, passing the live DB session.
    Call `.build()` to get a list of LangChain tool objects ready for the agent.
    """

    def __init__(self, db: Session):
        self.db = db
        self.station_repo = StationRepository()
        self.route_repo = RouteRepository()
        self.train_repo = TrainRepository()
        self.fare_repo = FareRepository()
        self.booking_repo = BookingRepository()
        self.timetable_svc = TimetableService()
        self.booking_svc = BookingService()
        self.fare_calc = FareCalculator()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve(self, name: str) -> str:
        """Resolve a station name/alias to a DB station code."""
        code = self.station_repo.fuzzy_find_station(name, self.db)
        return code or name.upper()

    # ------------------------------------------------------------------ #
    # Tool: search_trains                                                  #
    # ------------------------------------------------------------------ #

    def search_trains(self, source: str, destination: str, date: str = "") -> str:
        """
        Search for trains running between source and destination.

        Args:
            source: Source station name or code (e.g. 'Bangalore', 'SBC')
            destination: Destination station name or code (e.g. 'Mangalore', 'MAQ')
            date: Optional travel date in YYYY-MM-DD format (e.g. '2024-06-20')

        Returns:
            JSON string with list of trains or an error message.
        """
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            logger.info(f"[tool:search_trains] {src_code} → {dst_code} on {date or 'any date'}")
            trains = self.timetable_svc.search(src_code, dst_code, date or None, self.db)
            if not trains:
                return json.dumps({
                    "status": "no_results",
                    "message": f"No trains found from {source} to {destination}. "
                               "Try different station names or a different date."
                })
            result = []
            for t in trains[:10]:
                result.append({
                    "train_number": t.train_number,
                    "train_name": t.train_name,
                    "from": t.source_station_code,
                    "to": t.destination_station_code,
                })
            return json.dumps({
                "status": "ok",
                "count": len(trains),
                "shown": len(result),
                "source_resolved": src_code,
                "destination_resolved": dst_code,
                "trains": result,
            })
        except Exception as exc:
            logger.error(f"[tool:search_trains] Error: {exc}")
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------ #
    # Tool: get_fare                                                       #
    # ------------------------------------------------------------------ #

    def get_fare(
        self,
        train_number: str,
        source: str,
        destination: str,
        travel_class: str = "ALL",
        passengers: int = 1,
    ) -> str:
        """
        Get fare estimate for a train between two stations.

        Args:
            train_number: The train number (e.g. '16585')
            source: Source station name or code
            destination: Destination station name or code
            travel_class: Class code or name. Use 'ALL' to see all classes.
                          Options: GN, SL, 3A, 2A, 1A, CC, EC, 2S, ALL
            passengers: Number of passengers (default 1)

        Returns:
            JSON string with fare breakdown.
        """
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            cls_code = _norm_class(travel_class) if travel_class.upper() != "ALL" else "ALL"

            # Try to get the train name for category multiplier
            train = self.train_repo.get_by_number(train_number, self.db)
            train_name = train.train_name if train else ""

            # Try DB route distance first
            distance = self.route_repo.get_distance_between(
                train_number, src_code, dst_code, self.db
            )

            logger.info(f"[tool:get_fare] {train_number} {src_code}→{dst_code} "
                        f"dist={distance} class={cls_code} pax={passengers}")

            if cls_code == "ALL":
                fares_map = self.fare_calc.calculate_all_classes(
                    distance_km=distance,
                    train_name=train_name,
                    passengers=passengers,
                    source_code=src_code,
                    dest_code=dst_code,
                )
                output = {}
                for code, fb in fares_map.items():
                    output[code] = {
                        "class_name": fb.class_name,
                        "per_passenger": fb.per_passenger,
                        "total": fb.total_fare,
                        "distance_km": fb.distance_km,
                        "is_estimated": fb.is_estimated,
                    }
                return json.dumps({
                    "status": "ok",
                    "train_number": train_number,
                    "train_name": train_name,
                    "source": src_code,
                    "destination": dst_code,
                    "passengers": passengers,
                    "fares": output,
                })
            else:
                fb = self.fare_calc.calculate(
                    travel_class=cls_code,
                    distance_km=distance,
                    train_name=train_name,
                    passengers=passengers,
                    source_code=src_code,
                    dest_code=dst_code,
                )
                return json.dumps({
                    "status": "ok",
                    "train_number": train_number,
                    "train_name": train_name,
                    "source": src_code,
                    "destination": dst_code,
                    "class_code": fb.class_code,
                    "class_name": fb.class_name,
                    "distance_km": fb.distance_km,
                    "per_passenger": fb.per_passenger,
                    "total_fare": fb.total_fare,
                    "passengers": passengers,
                    "is_estimated": fb.is_estimated,
                })
        except Exception as exc:
            logger.error(f"[tool:get_fare] Error: {exc}")
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------ #
    # Tool: get_train_route                                                #
    # ------------------------------------------------------------------ #

    def get_train_route(self, train_number: str) -> str:
        """
        Get the full route/schedule for a specific train number.

        Args:
            train_number: The train number (e.g. '16585', '12627')

        Returns:
            JSON string with list of stops including times and distance.
        """
        try:
            logger.info(f"[tool:get_train_route] {train_number}")
            stops = self.route_repo.get_by_train(train_number, self.db)
            if not stops:
                return json.dumps({
                    "status": "no_results",
                    "message": f"No route data found for train {train_number}. Check the train number."
                })
            stops_data = []
            for s in stops:
                stops_data.append({
                    "seq": s.sequence,
                    "station_code": s.station_code,
                    "arrival": s.arrival_time or "--",
                    "departure": s.departure_time or "--",
                    "distance_km": s.distance_km,
                })
            return json.dumps({
                "status": "ok",
                "train_number": train_number,
                "total_stops": len(stops_data),
                "route": stops_data,
            })
        except Exception as exc:
            logger.error(f"[tool:get_train_route] Error: {exc}")
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------ #
    # Tool: book_ticket                                                    #
    # ------------------------------------------------------------------ #

    def book_ticket(
        self,
        source: str,
        destination: str,
        travel_class: str,
        passengers: int = 1,
        travel_date: str = "",
        train_number: str = "",
    ) -> str:
        """
        Create a simulated ticket booking.

        Args:
            source: Source station name or code
            destination: Destination station name or code
            travel_class: Class (e.g. SL, 3A, 2A, 1A, GN, CC)
            passengers: Number of passengers (default 1)
            travel_date: Date in YYYY-MM-DD format. Defaults to today.
            train_number: Optional specific train number to book on.

        Returns:
            JSON string with booking confirmation details.
        """
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            cls_code = _norm_class(travel_class)
            book_date = travel_date or date.today().isoformat()

            logger.info(f"[tool:book_ticket] {src_code}→{dst_code} "
                        f"cls={cls_code} pax={passengers} date={book_date}")

            entities = {
                "source_station": src_code,
                "destination_station": dst_code,
                "date": book_date,
                "travel_class": cls_code,
                "passengers": passengers,
                "train_number": train_number or None,
                "passenger_count": passengers,
                "class_type": cls_code,
            }
            booking = self.booking_svc.create_mock_booking(entities, self.db)

            # Compute fare for confirmation message
            dist = self.route_repo.get_distance_between(
                booking.train_number, src_code, dst_code, self.db
            )
            train = self.train_repo.get_by_number(booking.train_number, self.db)
            fare_breakdown = self.fare_calc.calculate(
                travel_class=cls_code,
                distance_km=dist,
                train_name=train.train_name if train else "",
                passengers=passengers,
                source_code=src_code,
                dest_code=dst_code,
            )

            return json.dumps({
                "status": "confirmed",
                "booking_id": booking.id,
                "train_number": booking.train_number,
                "source": src_code,
                "destination": dst_code,
                "class": cls_code,
                "passengers": passengers,
                "travel_date": book_date,
                "estimated_total_fare": fare_breakdown.total_fare,
                "per_passenger_fare": fare_breakdown.per_passenger,
            })
        except Exception as exc:
            logger.error(f"[tool:book_ticket] Error: {exc}")
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------ #
    # Tool: cancel_booking                                                 #
    # ------------------------------------------------------------------ #

    def cancel_booking(self, booking_id: int) -> str:
        """
        Cancel an existing booking by its booking ID.

        Args:
            booking_id: The numeric booking ID (e.g. 42)

        Returns:
            JSON string confirming cancellation or an error.
        """
        try:
            logger.info(f"[tool:cancel_booking] id={booking_id}")
            success = self.booking_repo.cancel(int(booking_id), self.db)
            if success:
                return json.dumps({
                    "status": "cancelled",
                    "booking_id": booking_id,
                    "message": f"Booking #{booking_id} has been successfully cancelled."
                })
            return json.dumps({
                "status": "not_found",
                "booking_id": booking_id,
                "message": f"Booking #{booking_id} was not found. Please check the ID."
            })
        except Exception as exc:
            logger.error(f"[tool:cancel_booking] Error: {exc}")
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------ #
    # Tool: get_booking_history                                            #
    # ------------------------------------------------------------------ #

    def get_booking_history(self, user_id: int = 1) -> str:
        """
        Retrieve booking history for the current user.

        Args:
            user_id: User ID (defaults to 1 for the demo session).

        Returns:
            JSON string with list of bookings.
        """
        try:
            logger.info(f"[tool:get_booking_history] user={user_id}")
            bookings = self.booking_repo.list_by_user(user_id, self.db)
            if not bookings:
                return json.dumps({
                    "status": "empty",
                    "message": "No bookings found for this user."
                })
            data = []
            for b in bookings:
                data.append({
                    "booking_id": b.id,
                    "train_number": b.train_number,
                    "class": b.travel_class,
                    "passengers": b.passenger_count,
                    "travel_date": str(b.travel_date),
                    "status": b.status,
                    "created_at": str(b.created_at),
                })
            return json.dumps({"status": "ok", "count": len(data), "bookings": data})
        except Exception as exc:
            logger.error(f"[tool:get_booking_history] Error: {exc}")
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------ #
    # Tool: get_station_info                                               #
    # ------------------------------------------------------------------ #

    def get_station_info(self, station: str) -> str:
        """
        Get information about a railway station.

        Args:
            station: Station name or code (e.g. 'Bangalore', 'SBC', 'Mangalore')

        Returns:
            JSON string with station details.
        """
        try:
            code = self._resolve(station)
            logger.info(f"[tool:get_station_info] {station} → {code}")
            s = self.station_repo.get_by_code(code, self.db)
            if not s:
                return json.dumps({
                    "status": "not_found",
                    "message": f"Station '{station}' not found. Try a different spelling or station code."
                })
            return json.dumps({
                "status": "ok",
                "station_code": s.station_code,
                "station_name": s.station_name,
                "city": s.city or "N/A",
            })
        except Exception as exc:
            logger.error(f"[tool:get_station_info] Error: {exc}")
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------ #
    # build() – returns LangChain @tool objects                           #
    # ------------------------------------------------------------------ #

    def build(self) -> list:
        """
        Return a list of LangChain-compatible tool callables, each with
        the correct docstring so the LLM knows how to call them.
        """
        instance = self

        @tool
        def search_trains(source: str, destination: str, date: str = "") -> str:
            """
            Search for trains running between source and destination.
            source: Source station name or code (e.g. 'Bangalore', 'SBC')
            destination: Destination station name or code (e.g. 'Mangalore', 'MAQ')
            date: Optional travel date in YYYY-MM-DD format
            """
            return instance.search_trains(source, destination, date)

        @tool
        def get_fare(
            train_number: str,
            source: str,
            destination: str,
            travel_class: str = "ALL",
            passengers: int = 1,
        ) -> str:
            """
            Get fare estimate for a train. Use travel_class='ALL' to see all classes.
            travel_class options: GN (General), SL (Sleeper), 3A, 2A, 1A, CC, EC, ALL
            """
            return instance.get_fare(train_number, source, destination, travel_class, passengers)

        @tool
        def get_train_route(train_number: str) -> str:
            """
            Get the full route and schedule for a specific train number.
            train_number: The train number e.g. '16585', '12627'
            """
            return instance.get_train_route(train_number)

        @tool
        def book_ticket(
            source: str,
            destination: str,
            travel_class: str,
            passengers: int = 1,
            travel_date: str = "",
            train_number: str = "",
        ) -> str:
            """
            Book a train ticket. Creates a simulated booking and returns confirmation.
            travel_class: SL, 3A, 2A, 1A, GN, CC, EC
            travel_date: YYYY-MM-DD format, defaults to today
            train_number: Optional specific train to book
            """
            return instance.book_ticket(
                source, destination, travel_class, passengers, travel_date, train_number
            )

        @tool
        def cancel_booking(booking_id: int) -> str:
            """
            Cancel an existing booking using its booking ID number.
            booking_id: The numeric booking ID shown in confirmation
            """
            return instance.cancel_booking(booking_id)

        @tool
        def get_booking_history(user_id: int = 1) -> str:
            """
            Get all previous bookings for the current user.
            user_id: Defaults to 1 (demo user)
            """
            return instance.get_booking_history(user_id)

        @tool
        def get_station_info(station: str) -> str:
            """
            Get details about a railway station by name or code.
            station: Station name or code e.g. 'Bangalore', 'SBC', 'Mangalore'
            """
            return instance.get_station_info(station)

        return [
            search_trains,
            get_fare,
            get_train_route,
            book_ticket,
            cancel_booking,
            get_booking_history,
            get_station_info,
        ]
