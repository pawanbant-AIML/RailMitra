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
the user asked, and flags uncertainty when the query depends on missing data.
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
        "howrah": "HWH",               # duplicate for kolkata already above, fine
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

    def __init__(self, station_aliases: Optional[Dict[str, str]] = None,
                 class_aliases: Optional[Dict[str, str]] = None,
                 confidence_threshold: float = 0.55,
                 preferred_max_results: int = 10) -> None:
        self.station_aliases = {**self.STATION_ALIASES, **(station_aliases or {})}
        self.class_aliases = {**self.CLASS_ALIASES, **(class_aliases or {})}
        self.confidence_threshold = confidence_threshold
        self.preferred_max_results = preferred_max_results
        try:
            from rapidfuzz import fuzz  # type: ignore
            self._rapidfuzz_ratio = fuzz.ratio
        except Exception:  # pragma: no cover
            self._rapidfuzz_ratio = None

    # ------------------------------------------------------------------
    # Main interpret entry point
    # ------------------------------------------------------------------
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
            "train_search", "fare_query", "booking_create", "booking_cancel",
            "booking_modify", "route_query", "train_info", "station_query"
        }
        clarification_question = (
            self._build_clarification_question(intent, missing_slots, slots)
            if clarification_needed else None
        )
        confidence = self._score_confidence(intent, slots, clarification_needed,
                                            normalized_text, previous_result)

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

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation except spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------
    def _detect_intent(self, text: str, previous_result: Optional[Dict[str, Any]] = None) -> str:
        if previous_result and previous_result.get("clarification_needed"):
            token_count = len(text.strip().split()) if text.strip() else 0
            if token_count <= 4:
                prev_intent = previous_result.get("intent")
                if prev_intent in {"booking_create", "fare_query", "train_search", "route_query"}:
                    return prev_intent

        if self._extract_train_number(text):
            return "train_info"

        scored = {intent: 0 for intent in self.INTENT_PATTERNS}
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in text:
                    scored[intent] += 1

        # Explicit preference hints → train_search even if vague
        if any(x in text for x in ("cheapest", "fastest", "best balance",
                                    "best train", "compare", "sort by fare",
                                    "sort by time", "fewest stops",
                                    "all available options", "lowest fare")):
            return "train_search"

        for intent in ("booking_cancel", "booking_modify", "booking_history", "booking_create"):
            if scored[intent] > 0:
                return intent
        if scored["fare_query"] > 0:
            return "fare_query"
        if scored["route_query"] > 0:
            return "route_query"
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

    # ------------------------------------------------------------------
    # Slot extractors
    # ------------------------------------------------------------------
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
            found_positions.sort(key=lambda x: x[0])
            codes = [c for _, c in found_positions]
            if len(codes) >= 2:
                return codes[0], codes[1]
            return codes[0], None
        return None, None

    def _resolve_station_phrase(self, phrase: str) -> Optional[str]:
        phrase = phrase.strip().lower()
        if phrase in self.station_aliases:
            return self.station_aliases[phrase]
        # Try fuzzy matching with station aliases
        best_code, best_score = None, 0
        for alias, code in self.station_aliases.items():
            score = self._string_similarity(phrase, alias)
            if score > best_score:
                best_score = score
                best_code = code
        return best_code if best_score >= 80 else None

    def _string_similarity(self, a: str, b: str) -> float:
        if self._rapidfuzz_ratio is not None:
            return self._rapidfuzz_ratio(a, b)
        return difflib.SequenceMatcher(None, a, b).ratio() * 100

    def _extract_train_number(self, text: str) -> Optional[str]:
        m = re.search(r"\b(\d{4,6})\b", text)
        return m.group(1) if m else None

    def _extract_class(self, text: str) -> Optional[str]:
        for alias, code in self.class_aliases.items():
            if alias in text:
                return code
        return None

    def _extract_passengers(self, text: str) -> Optional[int]:
        m = re.search(r"(\d+)\s*(passengers?|tickets?|people|persons)", text)
        if m:
            return int(m.group(1))
        for word, num in self.NUM_WORDS.items():
            if word in text:
                return num
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        today = date.today()
        if "today" in text:
            return today.isoformat()
        if "tomorrow" in text:
            return (today + timedelta(days=1)).isoformat()
        if "day after tomorrow" in text:
            return (today + timedelta(days=2)).isoformat()
        # Try explicit date patterns: DD/MM/YYYY, YYYY-MM-DD, "27th May"
        m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s*(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)", text, re.I)
        if m:
            day = int(m.group(1))
            month_str = m.group(2)[:3].lower()
            month_map = {
                "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12
            }
            month = month_map.get(month_str)
            if month:
                year = today.year
                return date(year, month, day).isoformat()
        return None

    def _extract_time_hint(self, text: str) -> Optional[str]:
        if any(w in text for w in ("morning", "early")):
            return "morning"
        if any(w in text for w in ("afternoon", "noon")):
            return "afternoon"
        if any(w in text for w in ("evening", "night", "late")):
            return "evening"
        return None

    def _extract_departure_after(self, text: str) -> Optional[str]:
        m = re.search(r"after\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", text, re.I)
        return m.group(1) if m else None

    def _extract_departure_before(self, text: str) -> Optional[str]:
        m = re.search(r"before\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", text, re.I)
        return m.group(1) if m else None

    def _extract_budget(self, text: str) -> Optional[int]:
        m = re.search(r"(?:under|less than|within|below|max|upto|<=)\s*(?:Rs\.?|INR|₹)?\s*(\d+)", text, re.I)
        if m:
            return int(m.group(1))
        m = re.search(r"budget\s*(?:of\s*)?(?:Rs\.?|INR|₹)?\s*(\d+)", text, re.I)
        if m:
            return int(m.group(1))
        return None

    def _extract_sort_hint(self, text: str) -> Optional[str]:
        if "cheapest" in text or "lowest fare" in text:
            return "fare"
        if "fastest" in text or "shortest" in text:
            return "duration"
        if "earliest" in text:
            return "departure"
        return None

    def _extract_limit(self, text: str) -> Optional[int]:
        m = re.search(r"top\s*(\d+)", text, re.I)
        if m:
            return int(m.group(1))
        return self.preferred_max_results

    def _extract_booking_id(self, text: str) -> Optional[str]:
        m = re.search(r"(?:pnr|booking\s*id)[\s:]*([a-z0-9]{6,})", text, re.I)
        return m.group(1).upper() if m else None

    def _extract_station_only(self, text: str) -> Optional[str]:
        # Heuristic: station info without src/dst pattern
        if re.search(r"(?:station|railway station|junction)", text, re.I):
            for alias, code in self.station_aliases.items():
                if alias in text:
                    return code
        return None

    def _extract_preference(self, text: str) -> Optional[str]:
        # Catch general preference: AC, non-AC, direct, etc.
        if "ac" in text and not any(c in text for c in ("1ac","2ac","3ac","1a","2a","3a")):
            return "ac"
        if "sleeper" in text:
            return "sleeper"
        return None

    # ------------------------------------------------------------------
    # Context merging (memory, previous result)
    # ------------------------------------------------------------------
    def _merge_context(
        self,
        slots: QuerySlots,
        memory: Dict[str, Any],
        previous_result: Optional[Dict[str, Any]],
        text: str,
    ) -> QuerySlots:
        # If previous result had search results and user said "first", "second", etc.
        if previous_result and previous_result.get("previous_results"):
            ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
            for word, idx in ordinals.items():
                if word in text:
                    slots.selected_option_index = idx
                    break

        # Fallback: use memory slots if current ones are missing
        mem_slots = memory.get("slots", {})
        if not slots.source and mem_slots.get("source"):
            slots.source = mem_slots["source"]
        if not slots.destination and mem_slots.get("destination"):
            slots.destination = mem_slots["destination"]
        if not slots.travel_date and mem_slots.get("travel_date"):
            slots.travel_date = mem_slots["travel_date"]
        if not slots.travel_class and mem_slots.get("travel_class"):
            slots.travel_class = mem_slots["travel_class"]

        # 🔥 NEW: also inherit from previous_result entities if still missing
        if previous_result:
            entities = previous_result.get("entities", {})
            if not slots.source and entities.get("source"):
                slots.source = entities["source"]
            if not slots.destination and entities.get("destination"):
                slots.destination = entities["destination"]
            if not slots.travel_date and entities.get("date"):
                slots.travel_date = entities["date"]
            if not slots.travel_class and entities.get("travel_class"):
                slots.travel_class = entities["travel_class"]
            if not slots.passengers and entities.get("passengers"):
                try:
                    slots.passengers = int(entities["passengers"])
                except (ValueError, TypeError):
                    pass

        return slots

    # ------------------------------------------------------------------
    # Slot completeness check
    # ------------------------------------------------------------------
    def _missing_slots_for_intent(self, intent: str, slots: QuerySlots,
                                  previous_result: Optional[Dict[str, Any]]) -> List[str]:
        missing: List[str] = []
        if intent in ("train_search", "fare_query", "route_query"):
            if not slots.source:
                missing.append("source")
            if not slots.destination:
                missing.append("destination")
            if intent == "fare_query" and not slots.travel_date:
                missing.append("travel_date")
        elif intent == "booking_create":
            if not slots.source:
                missing.append("source")
            if not slots.destination:
                missing.append("destination")
            if not slots.travel_date:
                missing.append("travel_date")
            if not slots.train_number and not previous_result:
                missing.append("train_number")
        elif intent in ("booking_cancel", "booking_modify"):
            if not slots.booking_id:
                missing.append("booking_id")
        elif intent == "train_info":
            if not slots.train_number:
                missing.append("train_number")
        elif intent == "station_query":
            if not slots.station:
                missing.append("station")
        return missing

    def _build_clarification_question(self, intent: str, missing: List[str],
                                      slots: QuerySlots) -> str:
        if "source" in missing and "destination" not in missing:
            return "From where would you like to travel?"
        if "destination" in missing and "source" not in missing:
            return "Where would you like to go?"
        if "source" in missing and "destination" in missing:
            return "Please tell me your source and destination stations."
        if "travel_date" in missing:
            return "On which date would you like to travel?"
        if "train_number" in missing:
            return "Which train number are you interested in?"
        if "booking_id" in missing:
            return "Please provide your booking ID (PNR)."
        return "Could you please provide more details?"

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------
    def _score_confidence(self, intent: str, slots: QuerySlots,
                          clarification_needed: bool, text: str,
                          previous_result: Optional[Dict[str, Any]]) -> float:
        # Start with a base confidence
        base = 0.85
        # Penalty for missing slots
        missing = self._missing_slots_for_intent(intent, slots, previous_result)
        base -= len(missing) * 0.15
        # Bonus if explicit date or train number given
        if slots.travel_date:
            base += 0.1
        if slots.train_number:
            base += 0.1
        if previous_result:
            base += 0.05
        base = max(0.0, min(1.0, base))
        return base

    def _build_notes(self, text: str, slots: QuerySlots,
                     memory: Dict[str, Any],
                     previous_result: Optional[Dict[str, Any]]) -> List[str]:
        notes: List[str] = []
        if not slots.source or not slots.destination:
            notes.append("Source or destination missing; may need user clarification.")
        if slots.selected_option_index is not None:
            notes.append(f"User selected option index {slots.selected_option_index} from previous results.")
        return notes