"""
agent/tools.py — LangChain-compatible tool definitions for RailMitra.

This version adds time-aware search and booking parameters so the agent can
handle requests like "after 8 PM", "evening train", and "overnight train".
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.repository.booking_repo import BookingRepository
from app.repository.fare_repo import FareRepository
from app.repository.route_repo import RouteRepository
from app.repository.station_repo import StationRepository
from app.repository.train_repo import TrainRepository
from app.services.booking_service import BookingService
from app.services.fare_calculator import FareCalculator
from app.services.recommendation_engine import RecommendationEngine
from app.services.timetable_service import TimetableService

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
    "yeshwanthpur": "YPR",
    "yeshwantpur": "YPR",
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
    if value.upper() == "ALL":
        return "ALL"
    return mapping.get(value.lower(), value.upper())


class AgentTools:
    """Instantiate once per request with the DB session."""

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
        self.recommendation_engine = RecommendationEngine()

    def _resolve(self, name: str) -> str:
        if not name:
            return ""
        raw = str(name).strip()
        try:
            code = self.station_repo.fuzzy_find_station(raw, self.db)
            if code:
                return str(code).upper()
        except Exception:
            pass
        lowered = raw.lower()
        for alias, code in _STATION_ALIASES.items():
            if alias in lowered:
                return code
        return raw.upper()

    def _parse_time(self, value: str) -> Optional[str]:
        if not value:
            return None
        v = str(value).strip()
        if not v:
            return None
        if len(v) == 5 and v[2] == ":":
            return v
        if len(v) == 4 and v.isdigit():
            return f"{v[:2]}:{v[2:]}"
        return v

    def _train_to_payload(self, train: Any) -> Dict[str, Any]:
        train_number = getattr(train, "train_number", None)
        source = getattr(train, "source_station_code", None)
        destination = getattr(train, "destination_station_code", None)
        route_dep = self.route_repo.get_departure_time(train_number, source, self.db) if train_number and source else None
        route_arr = self.route_repo.get_arrival_time(train_number, destination, self.db) if train_number and destination else None
        stops_count = getattr(train, "stops", None) or getattr(train, "total_stops", None)
        if stops_count is None and train_number:
            stops_count = self.route_repo.get_stop_count(train_number, self.db)
        return {
            "train_number": train_number,
            "train_name": getattr(train, "train_name", ""),
            "from": source,
            "to": destination,
            "departure": getattr(train, "departure_time", None) or route_dep,
            "arrival": getattr(train, "arrival_time", None) or route_arr,
            "duration": getattr(train, "duration", None) or getattr(train, "journey_time", None),
            "stops": stops_count,
            "note": getattr(train, "note", None) or "",
        }

    def search_trains(
        self,
        source: str,
        destination: str,
        date: str = "",
        departure_after: str = "",
        departure_before: str = "",
        time_hint: str = "",
        direct_only: bool = False,
        limit: int = 10,
    ) -> str:
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            logger.info("[tool:search_trains] %s → %s date=%s after=%s before=%s hint=%s", src_code, dst_code, date or "any", departure_after, departure_before, time_hint)
            trains = self.timetable_svc.search(
                src_code,
                dst_code,
                date or None,
                self.db,
                departure_after=self._parse_time(departure_after) or None,
                departure_before=self._parse_time(departure_before) or None,
                time_hint=time_hint or None,
                direct_only=direct_only,
                limit=max(1, min(int(limit or 10), 50)),
            )
            if not trains:
                return json.dumps({
                    "status": "no_results",
                    "message": f"No trains found from {source} to {destination}. Try different station names or time filters.",
                    "source_resolved": src_code,
                    "destination_resolved": dst_code,
                }, ensure_ascii=False)
            result = [self._train_to_payload(t) for t in trains[: max(1, min(int(limit or 10), 50))]]
            return json.dumps({
                "status": "ok",
                "count": len(trains),
                "shown": len(result),
                "source_resolved": src_code,
                "destination_resolved": dst_code,
                "trains": result,
            }, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[tool:search_trains] Error: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})

    def get_fare(
        self,
        train_number: str,
        source: str,
        destination: str,
        travel_class: str = "ALL",
        passengers: int = 1,
        travel_date: str = "",
        departure_after: str = "",
        departure_before: str = "",
        time_hint: str = "",
    ) -> str:
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            cls_code = _norm_class(travel_class)

            train = self.train_repo.get_by_number(train_number, self.db)
            train_name = getattr(train, "train_name", "") if train else ""

            distance = self.route_repo.get_distance_between(train_number, src_code, dst_code, self.db)
            logger.info("[tool:get_fare] %s %s→%s dist=%s class=%s pax=%s", train_number, src_code, dst_code, distance, cls_code, passengers)

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
                }, ensure_ascii=False)

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
            }, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[tool:get_fare] Error: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})

    def get_train_route(self, train_number: str) -> str:
        try:
            logger.info("[tool:get_train_route] %s", train_number)
            stops = self.route_repo.get_by_train(train_number, self.db)
            if not stops:
                return json.dumps({
                    "status": "no_results",
                    "message": f"No route data found for train {train_number}. Check the train number."
                })
            stops_data = []
            for s in stops:
                stops_data.append({
                    "seq": getattr(s, "sequence", None),
                    "station_code": getattr(s, "station_code", None),
                    "arrival": getattr(s, "arrival_time", None) or "--",
                    "departure": getattr(s, "departure_time", None) or "--",
                    "distance_km": getattr(s, "distance_km", None),
                })
            train = self.train_repo.get_by_number(train_number, self.db)
            return json.dumps({
                "status": "ok",
                "train_number": train_number,
                "train_name": getattr(train, "train_name", "") if train else "",
                "total_stops": len(stops_data),
                "route": stops_data,
            }, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[tool:get_train_route] Error: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})

    def book_ticket(
        self,
        source: str,
        destination: str,
        travel_class: str,
        passengers: int = 1,
        travel_date: str = "",
        train_number: str = "",
        departure_after: str = "",
        departure_before: str = "",
        time_hint: str = "",
    ) -> str:
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            cls_code = _norm_class(travel_class)
            book_date = travel_date or date.today().isoformat()

            logger.info("[tool:book_ticket] %s→%s cls=%s pax=%s date=%s", src_code, dst_code, cls_code, passengers, book_date)

            entities = {
                "source_station": src_code,
                "destination_station": dst_code,
                "date": book_date,
                "travel_class": cls_code,
                "passengers": passengers,
                "train_number": train_number or None,
                "passenger_count": passengers,
                "class_type": cls_code,
                "departure_after": departure_after or None,
                "departure_before": departure_before or None,
                "time_hint": time_hint or None,
            }
            booking = self.booking_svc.create_mock_booking(entities, self.db)

            dist = self.route_repo.get_distance_between(booking.train_number, src_code, dst_code, self.db)
            train = self.train_repo.get_by_number(booking.train_number, self.db)
            fare_breakdown = self.fare_calc.calculate(
                travel_class=cls_code,
                distance_km=dist,
                train_name=getattr(train, "train_name", "") if train else "",
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
            }, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[tool:book_ticket] Error: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})

    def cancel_booking(self, booking_id: int) -> str:
        try:
            logger.info("[tool:cancel_booking] id=%s", booking_id)
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
            logger.exception("[tool:cancel_booking] Error: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})

    def get_booking_history(self, user_id: int = 1) -> str:
        try:
            logger.info("[tool:get_booking_history] user=%s", user_id)
            bookings = self.booking_repo.list_by_user(user_id, self.db)
            if not bookings:
                return json.dumps({"status": "empty", "message": "No bookings found for this user."})
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
            return json.dumps({"status": "ok", "count": len(data), "bookings": data}, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[tool:get_booking_history] Error: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})

    def get_station_info(self, station: str) -> str:
        try:
            code = self._resolve(station)
            logger.info("[tool:get_station_info] %s → %s", station, code)
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
            }, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[tool:get_station_info] Error: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})

    def recommend_trains(
        self,
        source: str,
        destination: str,
        time_hint: str = "",
        departure_after: str = "",
        departure_before: str = "",
        travel_class: str = "SL",
        passengers: int = 1,
        preference: str = "",
        limit: int = 5,
    ) -> str:
        try:
            src_code = self._resolve(source)
            dst_code = self._resolve(destination)
            trains = self.timetable_svc.search(
                src_code,
                dst_code,
                None,
                self.db,
                departure_after=departure_after or None,
                departure_before=departure_before or None,
                time_hint=time_hint or None,
                direct_only=False,
                limit=max(1, min(int(limit or 5), 20)),
            )
            ranked = self.recommendation_engine.rank(
                trains,
                source=src_code,
                destination=dst_code,
                time_hint=time_hint or None,
                departure_after=departure_after or None,
                departure_before=departure_before or None,
                preference=preference or None,
                travel_class=travel_class,
                passengers=passengers,
                limit=limit,
            )
            payload = []
            for item in ranked:
                train = item.train if hasattr(item, "train") else item
                payload.append({
                    "train_number": getattr(train, "train_number", None) if not isinstance(train, dict) else train.get("train_number"),
                    "train_name": getattr(train, "train_name", "") if not isinstance(train, dict) else train.get("train_name", ""),
                    "score": getattr(item, "score", None),
                    "reasons": getattr(item, "reasons", []),
                })
            return json.dumps({
                "status": "ok",
                "source": src_code,
                "destination": dst_code,
                "count": len(payload),
                "trains": payload,
            }, ensure_ascii=False)
        except Exception as exc:
            logger.exception("[tool:recommend_trains] Error: %s", exc)
            return json.dumps({"status": "error", "message": str(exc)})

    def build(self) -> list:
        instance = self

        @tool
        def search_trains(
            source: str,
            destination: str,
            date: str = "",
            departure_after: str = "",
            departure_before: str = "",
            time_hint: str = "",
            direct_only: bool = False,
            limit: int = 10,
        ) -> str:
            return instance.search_trains(source, destination, date, departure_after, departure_before, time_hint, direct_only, limit)

        @tool
        def get_fare(
            train_number: str,
            source: str,
            destination: str,
            travel_class: str = "ALL",
            passengers: int = 1,
            travel_date: str = "",
            departure_after: str = "",
            departure_before: str = "",
            time_hint: str = "",
        ) -> str:
            return instance.get_fare(
                train_number,
                source,
                destination,
                travel_class,
                passengers,
                travel_date,
                departure_after,
                departure_before,
                time_hint,
            )

        @tool
        def get_train_route(train_number: str) -> str:
            return instance.get_train_route(train_number)

        @tool
        def book_ticket(
            source: str,
            destination: str,
            travel_class: str,
            passengers: int = 1,
            travel_date: str = "",
            train_number: str = "",
            departure_after: str = "",
            departure_before: str = "",
            time_hint: str = "",
        ) -> str:
            return instance.book_ticket(
                source,
                destination,
                travel_class,
                passengers,
                travel_date,
                train_number,
                departure_after,
                departure_before,
                time_hint,
            )

        @tool
        def cancel_booking(booking_id: int) -> str:
            return instance.cancel_booking(booking_id)

        @tool
        def get_booking_history(user_id: int = 1) -> str:
            return instance.get_booking_history(user_id)

        @tool
        def get_station_info(station: str) -> str:
            return instance.get_station_info(station)

        @tool
        def recommend_trains(
            source: str,
            destination: str,
            time_hint: str = "",
            departure_after: str = "",
            departure_before: str = "",
            travel_class: str = "SL",
            passengers: int = 1,
            preference: str = "",
            limit: int = 5,
        ) -> str:
            return instance.recommend_trains(
                source,
                destination,
                time_hint,
                departure_after,
                departure_before,
                travel_class,
                passengers,
                preference,
                limit,
            )

        return [
            search_trains,
            get_fare,
            get_train_route,
            book_ticket,
            cancel_booking,
            get_booking_history,
            get_station_info,
            recommend_trains,
        ]
