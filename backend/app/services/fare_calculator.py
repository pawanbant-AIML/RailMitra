"""
fare_calculator.py – Realistic Indian Railways fare estimation engine.

Design:
  - Per-km base rates for each class, calibrated to approximate real IRCTC fares.
  - Train category multipliers (Rajdhani/Shatabdi are premium; Express is standard).
  - Minimum fare floors to avoid unrealistically tiny amounts.
  - Graceful fallback when distance is unknown (uses route-average distances per class).

Class codes used across the system:
  GN  – General / Unreserved
  SL  – Sleeper Class
  3A  – AC 3-Tier
  2A  – AC 2-Tier
  1A  – AC First Class
  CC  – Chair Car (day trains)
  EC  – Executive Chair Car
  2S  – Second Sitting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Per-km base fares (₹ / km), calibrated for ~300–700 km corridors
# ---------------------------------------------------------------------------
BASE_RATE_PER_KM: Dict[str, float] = {
    "GN": 0.45,   # General – cheapest
    "2S": 0.50,   # Second Sitting
    "SL": 0.95,   # Sleeper – most popular
    "CC": 1.80,   # Chair Car (AC day train)
    "3A": 2.40,   # AC 3-Tier
    "2A": 3.60,   # AC 2-Tier
    "1A": 6.00,   # AC 1st Class
    "EC": 3.20,   # Executive Chair Car
}

# Minimum fare per class (₹) – even short journeys must charge at least this
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

# Train-name keyword → multiplier (applied on top of base rate)
# Rajdhani/Shatabdi charge a premium; Passenger trains are cheaper.
TRAIN_CATEGORY_MULTIPLIERS: Dict[str, float] = {
    "rajdhani": 1.45,
    "shatabdi": 1.35,
    "duronto": 1.40,
    "vande bharat": 1.50,
    "tejas": 1.50,
    "humsafar": 1.30,
    "garib rath": 0.90,
    "passenger": 0.65,
    "local": 0.55,
    "express": 1.00,
    "superfast": 1.10,
    "mail": 0.95,
    "intercity": 1.10,
}

# Fallback distances (km) used when route data is unavailable
# Keyed by (source_code, destination_code) pairs – major corridors only
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
    ("SBC", "GOA"): 580,
    ("GOA", "SBC"): 580,
    ("MAQ", "MAS"): 713,
    ("MAS", "MAQ"): 713,
    ("MYS", "MAS"): 488,
    ("MAS", "MYS"): 488,
}

# Default average distance used when nothing else is available
DEFAULT_DISTANCE_KM = 350


@dataclass
class FareBreakdown:
    """Structured output for a single class fare calculation."""
    class_code: str
    class_name: str
    distance_km: int
    base_fare: float
    multiplier: float
    final_fare: float
    per_passenger: float
    passengers: int
    total_fare: float
    is_estimated: bool  # True when distance was not from DB


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


class FareCalculator:
    """
    Calculates realistic per-km fare estimates for Indian Railways.

    Usage:
        calc = FareCalculator()
        breakdown = calc.calculate(
            travel_class="SL",
            distance_km=352,
            train_name="Karnataka Express",
            passengers=2
        )
        print(breakdown.total_fare)  # e.g. ₹686
    """

    def _get_train_multiplier(self, train_name: Optional[str]) -> float:
        """Returns the multiplier for the train category based on its name."""
        if not train_name:
            return 1.0
        lower = train_name.lower()
        for keyword, mult in TRAIN_CATEGORY_MULTIPLIERS.items():
            if keyword in lower:
                return mult
        return 1.0

    def _normalise_class(self, travel_class: str) -> str:
        """Normalise incoming class string to a known code."""
        mapping = {
            "sleeper": "SL", "sl": "SL",
            "general": "GN", "gn": "GN", "unreserved": "GN",
            "3ac": "3A", "3a": "3A", "third ac": "3A",
            "2ac": "2A", "2a": "2A", "second ac": "2A",
            "1ac": "1A", "1a": "1A", "first ac": "1A",
            "cc": "CC", "chair car": "CC", "ac chair": "CC",
            "ec": "EC", "executive": "EC",
            "2s": "2S", "second sitting": "2S",
        }
        return mapping.get(travel_class.lower().strip(), travel_class.upper().strip())

    def calculate(
        self,
        travel_class: str,
        distance_km: Optional[int],
        train_name: Optional[str] = None,
        passengers: int = 1,
        source_code: Optional[str] = None,
        dest_code: Optional[str] = None,
    ) -> FareBreakdown:
        """
        Calculate fare for a single class and return a FareBreakdown.

        Priority for distance:
          1. distance_km argument (from DB route data)
          2. CORRIDOR_DISTANCES lookup by station codes
          3. DEFAULT_DISTANCE_KM fallback
        """
        class_code = self._normalise_class(travel_class)
        is_estimated = False

        # --- Resolve distance ---
        if distance_km and distance_km > 0:
            dist = distance_km
        elif source_code and dest_code:
            key = (source_code.upper(), dest_code.upper())
            dist = CORRIDOR_DISTANCES.get(key, 0)
            if dist == 0:
                dist = DEFAULT_DISTANCE_KM
                is_estimated = True
            else:
                is_estimated = True  # Still an estimate (not from live DB)
        else:
            dist = DEFAULT_DISTANCE_KM
            is_estimated = True

        rate = BASE_RATE_PER_KM.get(class_code, BASE_RATE_PER_KM["SL"])
        multiplier = self._get_train_multiplier(train_name)
        raw_fare = dist * rate * multiplier
        floored = max(raw_fare, MIN_FARE.get(class_code, 100))
        # Round to nearest ₹5 for a realistic look
        final = round(floored / 5) * 5
        total = final * max(passengers, 1)

        return FareBreakdown(
            class_code=class_code,
            class_name=CLASS_DISPLAY_NAMES.get(class_code, class_code),
            distance_km=dist,
            base_fare=round(raw_fare, 2),
            multiplier=multiplier,
            final_fare=final,
            per_passenger=final,
            passengers=max(passengers, 1),
            total_fare=total,
            is_estimated=is_estimated,
        )

    def calculate_all_classes(
        self,
        distance_km: Optional[int],
        train_name: Optional[str] = None,
        passengers: int = 1,
        source_code: Optional[str] = None,
        dest_code: Optional[str] = None,
        available_classes: Optional[list] = None,
    ) -> Dict[str, FareBreakdown]:
        """
        Calculate fare for every class (or a specified subset).
        Returns a dict keyed by class code.
        """
        classes = available_classes or list(BASE_RATE_PER_KM.keys())
        result = {}
        for cls in classes:
            result[cls] = self.calculate(
                travel_class=cls,
                distance_km=distance_km,
                train_name=train_name,
                passengers=passengers,
                source_code=source_code,
                dest_code=dest_code,
            )
        return result

    def format_fare_table(
        self,
        source: str,
        destination: str,
        train_number: str,
        train_name: str,
        fares: Dict[str, FareBreakdown],
        passengers: int = 1,
    ) -> str:
        """Render a clean markdown fare table for the chat interface."""
        lines = [
            f"💰 **Fare Estimates — {source} → {destination}**",
            f"🚆 Train: **{train_number}** ({train_name})",
            f"👥 Passengers: **{passengers}**",
            "",
            "| Class | Per Ticket | Total |",
            "|-------|-----------|-------|",
        ]
        order = ["GN", "2S", "SL", "CC", "3A", "2A", "1A", "EC"]
        for cls in order:
            if cls in fares:
                f = fares[cls]
                lines.append(
                    f"| {f.class_name} | ₹{f.per_passenger:,.0f} | ₹{f.total_fare:,.0f} |"
                )
        if fares:
            sample = next(iter(fares.values()))
            note = "~estimated" if sample.is_estimated else "distance from route data"
            lines.append(f"\n_Fares are approximate ({note}, {sample.distance_km} km)._")
        return "\n".join(lines)
