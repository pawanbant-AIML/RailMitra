"""
fare_calculator.py — Fare estimation engine for RailMitra.

This file is designed to be resilient when route distance is missing in the
Datameet dataset:
- use live route distance if available
- otherwise fall back to corridor distance
- otherwise use a safe default distance
- return a structured breakdown plus a markdown table helper
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence


BASE_RATE_PER_KM: Dict[str, float] = {
    "GN": 0.45,
    "2S": 0.50,
    "SL": 0.95,
    "CC": 1.80,
    "3A": 2.40,
    "2A": 3.60,
    "1A": 6.00,
    "EC": 3.20,
}

MIN_FARE: Dict[str, float] = {
    "GN": 30,
    "2S": 35,
    "SL": 110,
    "CC": 200,
    "3A": 300,
    "2A": 450,
    "1A": 750,
    "EC": 400,
}

CLASS_DISPLAY_NAMES: Dict[str, str] = {
    "GN": "General / Unreserved",
    "2S": "Second Sitting",
    "SL": "Sleeper Class",
    "CC": "AC Chair Car",
    "3A": "AC 3-Tier",
    "2A": "AC 2-Tier",
    "1A": "AC First Class",
    "EC": "Executive Chair Car",
}

TRAIN_CATEGORY_MULTIPLIERS: Dict[str, float] = {
    "vande bharat": 1.50,
    "tejas": 1.50,
    "rajdhani": 1.45,
    "duronto": 1.40,
    "shatabdi": 1.35,
    "humsafar": 1.30,
    "superfast": 1.10,
    "intercity": 1.10,
    "express": 1.00,
    "mail": 0.95,
    "garib rath": 0.90,
    "passenger": 0.65,
    "local": 0.55,
}

SERVICE_CHARGES: Dict[str, int] = {
    "GN": 5,
    "2S": 10,
    "SL": 20,
    "CC": 25,
    "3A": 30,
    "2A": 35,
    "1A": 50,
    "EC": 40,
}

ROUND_TO = 5

CORRIDOR_DISTANCES: Dict[tuple, int] = {
    ("SBC", "MAQ"): 352,
    ("MAQ", "SBC"): 352,
    ("SBC", "MYS"): 139,
    ("MYS", "SBC"): 139,
    ("SBC", "UBL"): 400,
    ("UBL", "SBC"): 400,
    ("SBC", "MAS"): 361,
    ("MAS", "SBC"): 361,
    ("SBC", "NDLS"): 2444,
    ("NDLS", "SBC"): 2444,
    ("SBC", "HYB"): 574,
    ("HYB", "SBC"): 574,
    ("SBC", "PUNE"): 843,
    ("PUNE", "SBC"): 843,
    ("NDLS", "MAS"): 2175,
    ("MAS", "NDLS"): 2175,
    ("NDLS", "HWH"): 1450,
    ("HWH", "NDLS"): 1450,
    ("NDLS", "CSMT"): 1386,
    ("CSMT", "NDLS"): 1386,
    ("MAS", "HWH"): 1662,
    ("HWH", "MAS"): 1662,
    ("MAQ", "MAS"): 713,
    ("MAS", "MAQ"): 713,
    ("MYS", "MAS"): 488,
    ("MAS", "MYS"): 488,
    ("SBC", "GOA"): 580,
    ("GOA", "SBC"): 580,
}

DEFAULT_DISTANCE_KM = 350


@dataclass
class FareBreakdown:
    class_code: str
    class_name: str
    distance_km: int
    base_fare: float
    multiplier: float
    final_fare: float
    per_passenger: float
    passengers: int
    total_fare: float
    is_estimated: bool
    components: Dict[str, float]


class FareCalculator:
    def _normalise_class(self, travel_class: Optional[str]) -> str:
        if not travel_class:
            return "SL"
        value = str(travel_class).strip().lower()
        if value == "all":
            return "ALL"
        mapping = {
            "sleeper": "SL",
            "sl": "SL",
            "general": "GN",
            "gn": "GN",
            "unreserved": "GN",
            "2s": "2S",
            "second sitting": "2S",
            "3ac": "3A",
            "3a": "3A",
            "third ac": "3A",
            "2ac": "2A",
            "2a": "2A",
            "second ac": "2A",
            "1ac": "1A",
            "1a": "1A",
            "first ac": "1A",
            "cc": "CC",
            "chair car": "CC",
            "ec": "EC",
            "executive": "EC",
        }
        return mapping.get(value, value.upper())

    def _get_train_multiplier(self, train_name: Optional[str]) -> float:
        if not train_name:
            return 1.0
        lowered = str(train_name).lower()
        for keyword, mult in TRAIN_CATEGORY_MULTIPLIERS.items():
            if keyword in lowered:
                return mult
        return 1.0

    def _coerce_distance(self, distance_km: Optional[int], source_code: Optional[str], dest_code: Optional[str]) -> tuple[int, bool]:
        if isinstance(distance_km, (int, float)) and distance_km > 0:
            return int(round(distance_km)), False
        if source_code and dest_code:
            key = (str(source_code).upper(), str(dest_code).upper())
            if key in CORRIDOR_DISTANCES:
                return CORRIDOR_DISTANCES[key], True
        return DEFAULT_DISTANCE_KM, True

    def _round_rupees(self, value: float) -> int:
        if value <= 0:
            return 0
        return int(round(value / ROUND_TO) * ROUND_TO)

    def _fare_components(
        self,
        class_code: str,
        distance_km: int,
        train_name: Optional[str],
        passengers: int,
    ) -> Dict[str, float]:
        rate = BASE_RATE_PER_KM.get(class_code, BASE_RATE_PER_KM["SL"])
        train_mult = self._get_train_multiplier(train_name)
        service = SERVICE_CHARGES.get(class_code, 20)
        reservation = 0.0 if class_code in {"GN", "2S"} else (12 if class_code == "SL" else 20)

        base = float(distance_km) * rate
        premium = base * (train_mult - 1.0) if train_mult > 1 else 0.0
        subtotal = base + premium + service + reservation
        minimum = MIN_FARE.get(class_code, 100)
        final = max(subtotal, minimum)

        return {
            "rate_per_km": rate,
            "train_multiplier": train_mult,
            "service_charge": float(service),
            "reservation_charge": float(reservation),
            "base_amount": base,
            "premium_amount": premium,
            "subtotal": subtotal,
            "minimum_fare": float(minimum),
            "per_passenger": float(self._round_rupees(final)),
            "total": float(self._round_rupees(final) * max(passengers, 1)),
        }

    def calculate(
        self,
        travel_class: str,
        distance_km: Optional[int],
        train_name: Optional[str] = None,
        passengers: int = 1,
        source_code: Optional[str] = None,
        dest_code: Optional[str] = None,
    ) -> FareBreakdown:
        class_code = self._normalise_class(travel_class)
        if class_code == "ALL":
            class_code = "SL"

        distance, estimated = self._coerce_distance(distance_km, source_code, dest_code)
        passengers = max(1, int(passengers or 1))
        components = self._fare_components(class_code, distance, train_name, passengers)

        per_passenger = components["per_passenger"]
        total_fare = components["total"]

        return FareBreakdown(
            class_code=class_code,
            class_name=CLASS_DISPLAY_NAMES.get(class_code, class_code),
            distance_km=distance,
            base_fare=round(components["base_amount"], 2),
            multiplier=round(components["train_multiplier"], 2),
            final_fare=per_passenger,
            per_passenger=per_passenger,
            passengers=passengers,
            total_fare=total_fare,
            is_estimated=estimated,
            components=components,
        )

    def calculate_all_classes(
        self,
        distance_km: Optional[int],
        train_name: Optional[str] = None,
        passengers: int = 1,
        source_code: Optional[str] = None,
        dest_code: Optional[str] = None,
        available_classes: Optional[Sequence[str]] = None,
    ) -> Dict[str, FareBreakdown]:
        classes = list(available_classes) if available_classes else list(BASE_RATE_PER_KM.keys())
        output: Dict[str, FareBreakdown] = {}
        for cls in classes:
            normalized = self._normalise_class(cls)
            if normalized == "ALL":
                continue
            output[normalized] = self.calculate(
                travel_class=normalized,
                distance_km=distance_km,
                train_name=train_name,
                passengers=passengers,
                source_code=source_code,
                dest_code=dest_code,
            )
        return output

    def estimate_demo_fare(
        self,
        distance_km: float,
        class_code: str,
        train_category_multiplier: float = 1.0,
        passengers: int = 1,
    ) -> int:
        normalized = self._normalise_class(class_code)
        distance = max(1, int(round(distance_km)))
        rate = BASE_RATE_PER_KM.get(normalized, BASE_RATE_PER_KM["SL"])
        base = distance * rate * max(0.75, float(train_category_multiplier))
        final = max(base + SERVICE_CHARGES.get(normalized, 20), MIN_FARE.get(normalized, 100))
        return self._round_rupees(final) * max(1, int(passengers or 1))

    def format_fare_table(
        self,
        source: str,
        destination: str,
        train_number: str,
        train_name: str,
        fares: Dict[str, FareBreakdown],
        passengers: int = 1,
    ) -> str:
        lines = [
            f"💰 **Fare Estimates — {source} → {destination}**",
            f"🚆 Train: **{train_number}** ({train_name})",
            f"👥 Passengers: **{passengers}**",
            "",
            "| Class | Per Ticket | Total |",
            "|-------|-----------:|------:|",
        ]
        order = ["GN", "2S", "SL", "CC", "3A", "2A", "1A", "EC"]
        for cls in order:
            if cls in fares:
                f = fares[cls]
                lines.append(f"| {f.class_name} | ₹{f.per_passenger:,.0f} | ₹{f.total_fare:,.0f} |")
        if fares:
            sample = next(iter(fares.values()))
            note = "estimated" if sample.is_estimated else "route-based"
            lines.append(f"\n_Fares are approximate ({note}, {sample.distance_km} km)._")
        return "\n".join(lines)

    def summarize_breakdown(self, breakdown: FareBreakdown) -> str:
        return (
            f"{breakdown.class_name}: ₹{breakdown.per_passenger:,.0f} per passenger, "
            f"₹{breakdown.total_fare:,.0f} total for {breakdown.passengers} pax "
            f"({breakdown.distance_km} km, {'estimated' if breakdown.is_estimated else 'route-based'})."
        )
