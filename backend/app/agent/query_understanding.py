"""
agent/query_understanding.py

Research-grade intent, entity, and follow-up understanding for RailMitra.

Design goals:
- Handle natural language train queries, fare queries, route queries, booking actions,
  station questions, and multi-intent comparisons.
- Extract source/destination/train/class/passenger/date/time/budget preferences.
- Resolve common station aliases and spelling variations.
- Detect incomplete queries and produce a clarification payload.
- Return a confidence score plus slots so downstream services can choose
  between deterministic logic and LLM-assisted handling.
- Stay dependency-light: RapidFuzz is optional, not required.

This module is intentionally conservative around Datameet gaps:
it does not invent unavailable data. It only extracts and normalizes what
he user asked, and flags uncertainty when the query depends on missing data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
import difflib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class QuerySlots:
    source: Optional[str] = None
    destination: Optional[str] = None
    train_number: Optional[str] = None
    travel_class: Optional[str] = None
    passengers: Optional[int] = None
    travel_date: Optional[str] = None
    time_hint: Optional[str] = None
    departure_after: Optional[str] = None
    departure_before: Optional[str] = None
    budget_max: Optional[int] = None
    sort_by: Optional[str] = None
    limit: Optional[int] = None
    booking_id: Optional[str] = None
    station: Optional[str] = None
    preference: Optional[str] = None
    # For follow-ups like "book the first one" — 0-based index into previous results
    selected_option_index: Optional[int] = None


@dataclass
class QueryInterpretation:
    raw_text: str
    normalized_text: str
    intent: str
    sub_intents: List[str] = field(default_factory=list)
    slots: QuerySlots = field(default_factory=QuerySlots)
    missing_slots: List[str] = field(default_factory=list)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    confidence: float = 0.0
    resolved_entities: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "intent": self.intent,
            "sub_intents": list(self.sub_intents),
            "slots": asdict(self.slots),
            "missing_slots": list(self.missing_slots),
            "clarification_needed": self.clarification_needed,
            "clarification_question": self.clarification_question,
            "confidence": round(float(self.confidence), 4),
            "resolved_entities": dict(self.resolved_entities),
            "notes": list(self.notes),
        }


class QueryUnderstanding:
    """Parse rail-related natural language into a structured interpretation."""

    INTENT_PATTERNS: Dict[str, Sequence[str]] = {
        "booking_cancel": ("cancel booking", "cancel my booking", "reverse booking", "cancel ticket", "cancel"),
        "booking_modify": ("modify booking", "change booking", "edit booking", "change class", "upgrade", "downgrade"),
        "booking_history": ("my booking", "booking history", "my ticket", "my bookings", "reservations"),
        "booking_create": ("book", "reserve", "buy ticket", "reserve seat"),
        "fare_query": ("fare", "cost", "price", "how much", "charge", "estimate"),
        "route_query": ("route", "where does", "stop", "stations between", "journey", "duration"),
        "station_query": ("station", "station info", "tell me about", "near station", "station near"),
        "train_info": ("train number", "tell me about train", "details of train", "about train"),
        "train_search": ("show trains", "find trains", "available trains", "train from", "train to", "train between"),
        "greeting": ("hi", "hello", "hey", "namaste", "help"),
    }

    STATION_ALIASES: Dict[str, str] = {
        "bangalore": "SBC",
        "banglore": "SBC",
        "bangaluru": "SBC",
        "bengaluru": "SBC",
        "bangluru": "SBC",
        "blr": "SBC",
        "sbc": "SBC",
        "bengaluru city": "SBC",
        "bangalore city": "SBC",
        "yesvantpur": "YPR",
        "yeshwanthpur": "YPR",
        "yeshwantpur": "YPR",
        "ypr": "YPR",
        "kr puram": "KJM",
        "krishnarajapuram": "KJM",
        "mysore": "MYS",
        "mysuru": "MYS",
        "mys": "MYS",
        "mangalore": "MAQ",
        "mangaluru": "MAQ",
        "manglore": "MAQ",
        "mangluru": "MAQ",
        "mangaloor": "MAQ",
        "maq": "MAQ",
        "hubli": "UBL",
        "hubballi": "UBL",
        "ubl": "UBL",
        "udupi": "UD",
        "ud": "UD",
        "hassan": "HAS",
        "has": "HAS",
        "chennai": "MAS",
        "madras": "MAS",
        "mas": "MAS",
        "delhi": "NDLS",
        "delhii": "NDLS",
        "new delhi": "NDLS",
        "ndls": "NDLS",
        "mumbai": "CSMT",
        "bombay": "CSMT",
        "calcutta": "HWH",
        "kolkata": "HWH",
        "howrah": "HWH",
        "csmt": "CSMT",
        "goa": "MAO",
        "madgaon": "MAO",
        "mao": "MAO",
        "pune": "PUNE",
        "hyderabad": "HYB",
        "hydrabad": "HYB",
        "hyd": "HYB",
        "secunderabad": "SC",
        "coimbatore": "CBE",
        "ernakulam": "ERS",
        "kochi": "ERS",
        "trivandrum": "TVC",
        "thiruvananthapuram": "TVC",
        "kolkata": "HWH",
        "howrah": "HWH",
        "ahmedabad": "ADI",
        "surat": "ST",
        "vadodara": "BRC",
        "jaipur": "JP",
        "lucknow": "LKO",
        "patna": "PNBE",
        "bhopal": "BPL",
        "nagpur": "NGP",
        "indore": "INDB",
        "visakhapatnam": "VSKP",
        "vizag": "VSKP",
        "amritsar": "ASR",
        "chandigarh": "CDG",
        "guwahati": "GHY",
        "bhubaneswar": "BBS",
        "davangere": "DVG",
        "davanagere": "DVG",
    }

    CLASS_ALIASES: Dict[str, str] = {
        "general": "GN",
        "unreserved": "GN",
        "gn": "GN",
        "second sitting": "2S",
        "2s": "2S",
        "sleeper": "SL",
        "sl": "SL",
        "chair car": "CC",
        "cc": "CC",
        "3ac": "3A",
        "3 a": "3A",
        "3a": "3A",
        "third ac": "3A",
        "2ac": "2A",
        "2 a": "2A",
        "2a": "2A",
        "second ac": "2A",
        "1ac": "1A",
        "1 a": "1A",
        "1a": "1A",
        "first ac": "1A",
        "executive": "EC",
        "ec": "EC",
    }

    NUM_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    def __init__(self, station_aliases: Optional[Dict[str, str]] = None, class_aliases: Optional[Dict[str, str]] = None,
                 confidence_threshold: float = 0.55, preferred_max_results: int = 10) -> None:
        self.station_aliases = {**self.STATION_ALIASES, **(station_aliases or {})}
        self.class_aliases = {**self.CLASS_ALIASES, **(class_aliases or {})}
        self.confidence_threshold = confidence_threshold
        self.preferred_max_results = preferred_max_results
        try:
            from rapidfuzz import fuzz  # type: ignore
            self._rapidfuzz_ratio = fuzz.ratio
        except Exception:  # pragma: no cover
            self._rapidfuzz_ratio = None

    def interpret(self, text: str, memory: Optional[Dict[str, Any]] = None,
                  previous_result: Optional[Dict[str, Any]] = None) -> QueryInterpretation:
        raw_text = (text or "").strip()
        normalized_text = self._normalize(raw_text)
        memory = memory or {}
        previous_result = previous_result or {}

        intent = self._detect_intent(normalized_text, previous_result=previous_result)
        sub_intents = self._detect_sub_intents(normalized_text)

        slots = QuerySlots()
        slots.source, slots.destination = self._extract_stations(normalized_text)
        slots.train_number = self._extract_train_number(normalized_text)
        slots.travel_class = self._extract_class(normalized_text)
        slots.passengers = self._extract_passengers(normalized_text)
        slots.travel_date = self._extract_date(normalized_text)
        slots.time_hint = self._extract_time_hint(normalized_text)
        slots.departure_after = self._extract_departure_after(normalized_text)
        slots.departure_before = self._extract_departure_before(normalized_text)
        slots.budget_max = self._extract_budget(normalized_text)
        slots.sort_by = self._extract_sort_hint(normalized_text)
        slots.limit = self._extract_limit(normalized_text)
        slots.booking_id = self._extract_booking_id(normalized_text)
        slots.station = self._extract_station_only(normalized_text)
        slots.preference = self._extract_preference(normalized_text)

        slots = self._merge_context(slots, memory, previous_result, normalized_text)
        missing_slots = self._missing_slots_for_intent(intent, slots, previous_result)
        clarification_needed = bool(missing_slots) and intent in {
            "train_search", "fare_query", "booking_create", "booking_cancel", "booking_modify", "route_query", "train_info", "station_query"
        }
        clarification_question = self._build_clarification_question(intent, missing_slots, slots) if clarification_needed else None
        confidence = self._score_confidence(intent, slots, clarification_needed, normalized_text, previous_result)

        resolved_entities = {
            "source": slots.source,
            "destination": slots.destination,
            "train_number": slots.train_number,
            "travel_class": slots.travel_class,
            "station": slots.station,
            "preference": slots.preference,
        }

        notes = self._build_notes(normalized_text, slots, memory, previous_result)

        return QueryInterpretation(
            raw_text=raw_text,
            normalized_text=normalized_text,
            intent=intent,
            sub_intents=sub_intents,
            slots=slots,
            missing_slots=missing_slots,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            confidence=confidence,
            resolved_entities=resolved_entities,
            notes=notes,
        )

    def _detect_intent(self, text: str, previous_result: Optional[Dict[str, Any]] = None) -> str:
        # If the previous result was asking for clarification for a particular intent,
        # short follow-up replies should inherit that intent (e.g., 'Tomorrow', '3A', '2').
        if previous_result and previous_result.get("clarification_needed"):
            token_count = len(text.strip().split()) if text.strip() else 0
            if token_count <= 4:
                prev_intent = previous_result.get("intent")
                if prev_intent in {"booking_create", "fare_query", "train_search", "route_query"}:
                    return prev_intent

        # If a train number is present, prefer train_info (explicit train queries)
        if self._extract_train_number(text):
            return "train_info"

        scored = {intent: 0 for intent in self.INTENT_PATTERNS}
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in text:
                    scored[intent] += 1

        if any(x in text for x in ("cheapest", "fastest", "best balance", "best train", "compare", "sort by fare", "sort by time", "fewest stops", "all available options", "lowest fare")):
            return "train_search"

        for intent in ("booking_cancel", "booking_modify", "booking_history", "booking_create"):
            if scored[intent] > 0:
                return intent
        if scored["fare_query"] > 0:
            return "fare_query"
        if scored["route_query"] > 0:
            return "route_query"
        # Prefer train_info over generic station_query when phrasing could match both
        if scored["train_info"] > 0:
            return "train_info"
        if scored["station_query"] > 0:
            return "station_query"
        if scored["train_search"] > 0:
            return "train_search"
        if scored["greeting"] > 0:
            return "greeting"
        return "train_search"

    def _detect_sub_intents(self, text: str) -> List[str]:
        sub_intents: List[str] = []
        for label, words in (
            ("cheapest", ("cheapest", "lowest fare", "lowest price", "minimum cost")),
            ("fastest", ("fastest", "least time", "shortest journey", "quickest")),
            ("direct_only", ("direct only", "non stop", "non-stop", "without change")),
            ("fewest_stops", ("fewest stops", "least stops", "fewer stops")),
            ("compare", ("compare", "comparison", "vs", "versus")),
            ("overnight", ("overnight", "night train", "tonight")),
        ):
            if any(w in text for w in words):
                sub_intents.append(label)
        return sub_intents

    def _extract_stations(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        patterns = [
            r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+tomorrow|\s+today|\s+tonight|\s+after|\s+before|\s+at|$)",
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+tomorrow|\s+today|\s+tonight|\s+after|\s+before|\s+at|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                src_phrase = m.group(1).strip()
                dst_phrase = m.group(2).strip()
                # remove trailing 'via ...' parts which can interfere with direct destination
                dst_phrase = re.split(r"\s+via\b", dst_phrase, flags=re.IGNORECASE)[0].strip()
                src_phrase = re.split(r"\s+via\b", src_phrase, flags=re.IGNORECASE)[0].strip()
                src = self._resolve_station_phrase(src_phrase)
                dst = self._resolve_station_phrase(dst_phrase)
                return src, dst

        found_positions: List[Tuple[int, str]] = []
        padded = f" {text} "
        for alias, code in self.station_aliases.items():
            idx = padded.find(f" {alias} ")
            if idx != -1:
                found_positions.append((idx, code))
        if found_positions:
            # sort by occurrence in text to preserve order
            found_positions.sort(key=lambda x: x[0])
            codes = [c for _, c in found_positions]
            if len(codes) >= 2:
                return codes[0], codes[1]
            return codes[0], None
        return None, None

        
