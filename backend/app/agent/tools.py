"""
agent/tools.py — LangChain-compatible tool definitions for RailMitra.

This version is intentionally conservative:
- JSON output for every tool
- Strong station resolution
- More fields returned for search/train/route/fare
- Graceful handling of incomplete Datameet railway coverage
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.repository.booking_repo import BookingRepository
from app.repository.fare_repo import FareRepository
from app.repository.route_repo import RouteRepository
from app.repository.station_repo import StationRepository
from app.repository.train_repo import TrainRepository
from app.services.booking_service import BookingService
from app.services.fare_calculator import FareCalculator
from app.services.timetable_service import TimetableService
from app.core.logger import logger

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

_STATION_ALIASES = {
    "bangalore": "SBC",
    "bengaluru": "SBC",
    "blr": "SBC",
    "sbc": "SBC",
    "yesvantpur": "YPR",
    "ypr": "YPR",
    "mysore": "MYS",
    "mysuru": "MYS",
    "mys": "MYS",
    "hubli": "UBL",
    "hubballi": "UBL",
    "ubl": "UBL",
    "mangalore": "MAQ",
    "mangaluru": "MAQ",
    "maq": "MAQ",
    "mumbai": "CSMT",
    "bombay": "CSMT",
    "csmt": "CSMT",
    "delhi": "NDLS",
    "new delhi": "NDLS",
    "ndls": "NDLS",
    "chennai": "MAS",
    "madras": "MAS",
    "mas": "MAS",
    "kolkata": "HWH",
    "howrah": "HWH",
    "hwh": "HWH",
    "hyderabad": "HYB",
    "hyd": "HYB",
    "secunderabad": "SC",
    "pune": "PUNE",
    "goa": "MAO",
    "udupi": "UD",
    "hassan": "HAS",
    "shivamogga": "SMET",
    "shimoga": "SMET",
    "davangere": "DVG",
    "kochi": "ERS",
    "ernakulam": "ERS",
    "ers": "ERS",
    "trivandrum": "TVC",
    "thiruvananthapuram": "TVC",
    "tvc": "TVC",
    "coimbatore": "CBE",
    "madurai": "MDU",
    "ahmedabad": "ADI",
    "surat": "ST",
    "vadodara": "BRC",
    "jaipur": "JP",
    "jodhpur": "JU",
    "udaipur": "UDZ",
    "lucknow": "LKO",
    "varanasi": "BSB",
    "patna": "PNBE",
    "bhopal": "BPL",
    "nagpur": "NGP",
    "indore": "INDB",
    "visakhapatnam": "VSKP",
    "vizag": "VSKP",
    "amritsar": "ASR",
    "chandigarh": "CDG",
    "ludhiana": "LDH",
    "guwahati": "GHY",
    "bhubaneswar": "BBS",
}


def _norm_class(cls: str) -> str:
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
    value = (cls or "").strip()
    if not value:
        return "SL"
    return mapping.get(value.lower(), value.upper())


class AgentTools:
    """
    Instantiate once per request with the DB session.
    Returns LangChain-compatible tools via `.build()`.
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, name: str) -> str:
        raw = (name or "").strip()
        if not raw:
            return raw
        code = self.station_repo.fuzzy_find_station(raw, self.db)
        if code:
            return code
        cleaned = re.sub(r"\s+", " ", raw.lower()).strip()
        if cleaned in _STATION_ALIASES:
            return _STATION_ALIASES[cleaned]
        for alias, alias_code in _STATION_ALIASES.items():
            if alias in cleaned:
                return alias_code
        return raw.upper()

    def _safe_attr(self, obj: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(obj, dict) and name in obj:
                val = obj.get(name)
                if val is not None:
                    return val
            if hasattr(obj, name):
                val = getattr(obj, name)
                if val is not None:
                    return val
        return default

    def _serialize_train(self, train: Any) -> Dict[str, Any]:
        return {
            "train_number": self._safe_attr(train, "train_number", "number", default=""),
            "train_name": self._safe_attr(train, "train_name", "name", default=""),
            "source": self._safe_attr(train, "source_station_code", "source", "from_station", default=""),
            "destination": self._safe_attr(train, "destination_station_code", "destination", "to_station", default=""),
            "departure": self._safe_attr(train, "departure_time", "departure", "dep", default=""),
            "arrival": self._safe_attr(train, "arrival_time", "arrival", "arr", default=""),
            "duration": self._safe_attr(train, "duration", "journey_time", "travel_time", default=""),
            "stops": self._safe_attr(train, "stops", "total_stops", "stop_count", default=None),
        }

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def search_trains(self, source: str, destination: str, date: str = "", limit: int = 20) -> str:
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            logger.info(f"[tool:search_trains] {src_code} → {dst_code} on {date or 'any date'}")

            trains = self.timetable_svc.search(src_code, dst_code, date or None, self.db) or []
            if not trains:
                return json.dumps({
                    "status": "no_results",
                    "message": f"No trains found from {source} to {destination}. Try different station names or a different date.",
                    "source_resolved": src_code,
                    "destination_resolved": dst_code,
                    "trains": [],
                })

            serialized = [self._serialize_train(t) for t in trains[: max(1, int(limit))]]
            return json.dumps({
                "status": "ok",
                "count": len(trains),
                "shown": len(serialized),
                "source_resolved": src_code,
                "destination_resolved": dst_code,
                "date": date or "",
                "trains": serialized,
            })
        except Exception as exc:
            logger.exception("[tool:search_trains] Error")
            return json.dumps({"status": "error", "message": str(exc)})

    def get_train_info(self, train_number: str) -> str:
        try:
            train = self.train_repo.get_by_number(train_number, self.db)
            if not train:
                return json.dumps({
                    "status": "not_found",
                    "message": f"Train {train_number} was not found in the available data.",
                })

            payload = {
                "status": "ok",
                "train_number": self._safe_attr(train, "train_number", "number", default=train_number),
                "train_name": self._safe_attr(train, "train_name", "name", default=""),
                "source": self._safe_attr(train, "source_station_code", "source", default=""),
                "destination": self._safe_attr(train, "destination_station_code", "destination", default=""),
                "days": self._safe_attr(train, "running_days", "days", "day_pattern", default=""),
                "duration": self._safe_attr(train, "duration", "journey_time", "travel_time", default=""),
                "type": self._safe_attr(train, "train_type", "type", default=""),
            }
            return json.dumps(payload)
        except Exception as exc:
            logger.exception("[tool:get_train_info] Error")
            return json.dumps({"status": "error", "message": str(exc)})

    def get_fare(
        self,
        train_number: str,
        source: str,
        destination: str,
        travel_class: str = "ALL",
        passengers: int = 1,
    ) -> str:
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            cls_code = _norm_class(travel_class) if travel_class.upper() != "ALL" else "ALL"

            train = self.train_repo.get_by_number(train_number, self.db)
            train_name = self._safe_attr(train, "train_name", "name", default="") if train else ""

            distance = self.route_repo.get_distance_between(train_number, src_code, dst_code, self.db)
            logger.info(f"[tool:get_fare] {train_number} {src_code}→{dst_code} dist={distance} class={cls_code} pax={passengers}")

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
                        "class_code": getattr(fb, "class_code", code),
                        "class_name": getattr(fb, "class_name", _CLASS_NAMES.get(code, code)),
                        "per_passenger": getattr(fb, "per_passenger", 0),
                        "total": getattr(fb, "total_fare", 0),
                        "distance_km": getattr(fb, "distance_km", distance),
                        "is_estimated": getattr(fb, "is_estimated", True),
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
                "class_code": getattr(fb, "class_code", cls_code),
                "class_name": getattr(fb, "class_name", _CLASS_NAMES.get(cls_code, cls_code)),
                "distance_km": getattr(fb, "distance_km", distance),
                "per_passenger": getattr(fb, "per_passenger", 0),
                "total_fare": getattr(fb, "total_fare", 0),
                "passengers": passengers,
                "is_estimated": getattr(fb, "is_estimated", True),
            })
        except Exception as exc:
            logger.exception("[tool:get_fare] Error")
            return json.dumps({"status": "error", "message": str(exc)})

    def get_train_route(self, train_number: str) -> str:
        try:
            logger.info(f"[tool:get_train_route] {train_number}")
            stops = self.route_repo.get_by_train(train_number, self.db) or []
            if not stops:
                return json.dumps({
                    "status": "no_results",
                    "message": f"No route data found for train {train_number}. Check the train number.",
                })
            route = []
            for s in stops:
                route.append({
                    "seq": self._safe_attr(s, "sequence", "seq", default=None),
                    "station_code": self._safe_attr(s, "station_code", "station", default=""),
                    "arrival": self._safe_attr(s, "arrival_time", "arrival", default="--"),
                    "departure": self._safe_attr(s, "departure_time", "departure", default="--"),
                    "distance_km": self._safe_attr(s, "distance_km", default=None),
                })
            return json.dumps({
                "status": "ok",
                "train_number": train_number,
                "total_stops": len(route),
                "route": route,
            })
        except Exception as exc:
            logger.exception("[tool:get_train_route] Error")
            return json.dumps({"status": "error", "message": str(exc)})

    def book_ticket(
        self,
        source: str,
        destination: str,
        travel_class: str,
        passengers: int = 1,
        travel_date: str = "",
        train_number: str = "",
    ) -> str:
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            cls_code = _norm_class(travel_class)
            book_date = travel_date or date.today().isoformat()

            logger.info(f"[tool:book_ticket] {src_code}→{dst_code} cls={cls_code} pax={passengers} date={book_date}")

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

            dist = self.route_repo.get_distance_between(booking.train_number, src_code, dst_code, self.db)
            train = self.train_repo.get_by_number(booking.train_number, self.db)
            train_name = self._safe_attr(train, "train_name", "name", default="") if train else ""

            fare_breakdown = self.fare_calc.calculate(
                travel_class=cls_code,
                distance_km=dist,
                train_name=train_name,
                passengers=passengers,
                source_code=src_code,
                dest_code=dst_code,
            )

            return json.dumps({
                "status": "confirmed",
                "booking_id": self._safe_attr(booking, "id", "booking_id", default=None),
                "train_number": self._safe_attr(booking, "train_number", default=train_number),
                "source": src_code,
                "destination": dst_code,
                "class": cls_code,
                "passengers": passengers,
                "travel_date": book_date,
                "estimated_total_fare": self._safe_attr(fare_breakdown, "total_fare", default=0),
                "per_passenger_fare": self._safe_attr(fare_breakdown, "per_passenger", default=0),
            })
        except Exception as exc:
            logger.exception("[tool:book_ticket] Error")
            return json.dumps({"status": "error", "message": str(exc)})

    def cancel_booking(self, booking_id: int) -> str:
        try:
            logger.info(f"[tool:cancel_booking] id={booking_id}")
            success = self.booking_repo.cancel(int(booking_id), self.db)
            if success:
                return json.dumps({
                    "status": "cancelled",
                    "booking_id": booking_id,
                    "message": f"Booking #{booking_id} has been successfully cancelled.",
                })
            return json.dumps({
                "status": "not_found",
                "booking_id": booking_id,
                "message": f"Booking #{booking_id} was not found. Please check the ID.",
            })
        except Exception as exc:
            logger.exception("[tool:cancel_booking] Error")
            return json.dumps({"status": "error", "message": str(exc)})

    def get_booking_history(self, user_id: int = 1) -> str:
        try:
            logger.info(f"[tool:get_booking_history] user={user_id}")
            bookings = self.booking_repo.list_by_user(user_id, self.db) or []
            if not bookings:
                return json.dumps({
                    "status": "empty",
                    "message": "No bookings found for this user.",
                })
            data = []
            for b in bookings:
                data.append({
                    "booking_id": self._safe_attr(b, "id", "booking_id", default=None),
                    "train_number": self._safe_attr(b, "train_number", default=""),
                    "class": self._safe_attr(b, "travel_class", "class", default=""),
                    "passengers": self._safe_attr(b, "passenger_count", "passengers", default=1),
                    "travel_date": str(self._safe_attr(b, "travel_date", default="")),
                    "status": self._safe_attr(b, "status", default=""),
                    "created_at": str(self._safe_attr(b, "created_at", default="")),
                })
            return json.dumps({"status": "ok", "count": len(data), "bookings": data})
        except Exception as exc:
            logger.exception("[tool:get_booking_history] Error")
            return json.dumps({"status": "error", "message": str(exc)})

    def get_station_info(self, station: str) -> str:
        try:
            code = self._resolve(station)
            logger.info(f"[tool:get_station_info] {station} → {code}")
            s = self.station_repo.get_by_code(code, self.db)
            if not s:
                return json.dumps({
                    "status": "not_found",
                    "message": f"Station '{station}' not found. Try a different spelling or station code.",
                })
            return json.dumps({
                "status": "ok",
                "station_code": self._safe_attr(s, "station_code", default=code),
                "station_name": self._safe_attr(s, "station_name", default=station),
                "city": self._safe_attr(s, "city", default="N/A"),
                "station_type": self._safe_attr(s, "station_type", default="Railway station"),
            })
        except Exception as exc:
            logger.exception("[tool:get_station_info] Error")
            return json.dumps({"status": "error", "message": str(exc)})

    # ------------------------------------------------------------------
    # Tool export
    # ------------------------------------------------------------------

    def build(self) -> list:
        instance = self

        @tool
        def search_trains(source: str, destination: str, date: str = "", limit: int = 20) -> str:
            """Search for trains between source and destination."""
            return instance.search_trains(source, destination, date, limit)

        @tool
        def get_train_info(train_number: str) -> str:
            """Get summary information for a specific train number."""
            return instance.get_train_info(train_number)

        @tool
        def get_fare(
            train_number: str,
            source: str,
            destination: str,
            travel_class: str = "ALL",
            passengers: int = 1,
        ) -> str:
            """Get fare estimate for a train. Use travel_class='ALL' to see all classes."""
            return instance.get_fare(train_number, source, destination, travel_class, passengers)

        @tool
        def get_train_route(train_number: str) -> str:
            """Get the full route and schedule for a specific train number."""
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
            """Book a train ticket and return a simulated confirmation."""
            return instance.book_ticket(source, destination, travel_class, passengers, travel_date, train_number)

        @tool
        def cancel_booking(booking_id: int) -> str:
            """Cancel an existing booking using its booking ID number."""
            return instance.cancel_booking(booking_id)

        @tool
        def get_booking_history(user_id: int = 1) -> str:
            """Get booking history for the current user."""
            return instance.get_booking_history(user_id)

        @tool
        def get_station_info(station: str) -> str:
            """Get details about a railway station by name or code."""
            return instance.get_station_info(station)

        return [
            search_trains,
            get_train_info,
            get_fare,
            get_train_route,
            book_ticket,
            cancel_booking,
            get_booking_history,
            get_station_info,
        ]
