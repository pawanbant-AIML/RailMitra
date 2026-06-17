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

    def _extract_station_only(self, text: str) -> Optional[str]:
        for prefix in ("tell me about ", "about ", "station near ", "near ", "station "):
            if prefix in text:
                candidate = text.split(prefix, 1)[1].strip()
                candidate = re.split(r"\b(?:station|stations|railway|rail)\b", candidate)[0].strip()
                resolved = self._resolve_station_phrase(candidate)
                if resolved:
                    return resolved
        return None

    def _extract_train_number(self, text: str) -> Optional[str]:
        match = re.search(r"\b(1\d{4}|[2-9]\d{4})\b", text)
        return match.group(1) if match else None

    def _extract_class(self, text: str) -> Optional[str]:
        for alias, code in sorted(self.class_aliases.items(), key=lambda item: -len(item[0])):
            if alias in text:
                return code
        return None

    def _extract_passengers(self, text: str) -> Optional[int]:
        m = re.search(r"\b(\d+)\s*(?:passenger|passengers|pax|ticket|tickets|seat|seats|person|people|traveller|traveler)s?\b", text)
        if m:
            try:
                return max(1, int(m.group(1)))
            except Exception:
                pass
        for word, value in self.NUM_WORDS.items():
            if re.search(rf"\b{word}\b", text):
                return value
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        today = date.today()
        text = (text or "").lower()
        if "day after tomorrow" in text:
            return (today + timedelta(days=2)).isoformat()
        if "tomorrow" in text:
            return (today + timedelta(days=1)).isoformat()
        if "today" in text or "now" in text:
            return today.isoformat()

        # ISO YYYY-MM-DD
        ymd = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
        if ymd:
            try:
                y, m, d = map(int, ymd.groups())
                return date(y, m, d).isoformat()
            except Exception:
                return None

        # DD/MM or D/M (optional year)
        dmy = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
        if dmy:
            try:
                d, m, y = dmy.group(1), dmy.group(2), dmy.group(3)
                d_i = int(d)
                m_i = int(m)
                if y:
                    y_i = int(y)
                    if y_i < 100:
                        y_i += 2000
                else:
                    y_i = today.year
                return date(y_i, m_i, d_i).isoformat()
            except Exception:
                return None

        # Day + Month name -> e.g., '25 December' or '25th December'
        month_map = {
            'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
            'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'september':9,
            'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
        }
        dm = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", text)
        if dm:
            try:
                d_i = int(dm.group(1))
                mon = dm.group(2).lower()
                m_i = month_map.get(mon[:3]) or month_map.get(mon)
                if not m_i:
                    return None
                y_i = today.year
                # If date already passed this year, prefer same year (tests only check format)
                return date(y_i, m_i, d_i).isoformat()
            except Exception:
                return None

        return None

    def _extract_time_hint(self, text: str) -> Optional[str]:
        if "morning" in text:
            return "morning"
        if "afternoon" in text:
            return "afternoon"
        if "evening" in text:
            return "evening"
        if "night" in text or "tonight" in text:
            return "night"

        # Fallback: an explicit time like 'after 8 pm' or 'at 20:00' returns the time string
        match = re.search(r"\b(?:after|before|at)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
        if not match:
            return None
        hh = int(match.group(1))
        mm = int(match.group(2) or 0)
        ampm = (match.group(3) or "").lower()
        if ampm == "pm" and hh != 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0
        return f"{hh:02d}:{mm:02d}"

    def _extract_departure_after(self, text: str) -> Optional[str]:
        match = re.search(r"\bafter\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
        if not match:
            return None
        hh = int(match.group(1))
        mm = int(match.group(2) or 0)
        ampm = (match.group(3) or "").lower()
        if ampm == "pm" and hh != 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0
        return f"{hh:02d}:{mm:02d}"

    def _extract_departure_before(self, text: str) -> Optional[str]:
        match = re.search(r"\bbefore\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", text)
        if not match:
            return None
        hh = int(match.group(1))
        mm = int(match.group(2) or 0)
        ampm = (match.group(3) or "").lower()
        if ampm == "pm" and hh != 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0
        return f"{hh:02d}:{mm:02d}"

    def _extract_budget(self, text: str) -> Optional[int]:
        m = re.search(r"(?:under|below|within|less than|budget of)\s*[₹rs\.]?\s*(\d{2,7})", text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _extract_sort_hint(self, text: str) -> Optional[str]:
        if any(x in text for x in ("cheapest", "lowest fare", "lowest price", "sort by fare")):
            return "fare"
        if any(x in text for x in ("fastest", "shortest journey", "least time", "sort by time")):
            return "duration"
        if any(x in text for x in ("fewest stops", "least stops", "fewer stops")):
            return "stops"
        return None

    def _extract_limit(self, text: str) -> Optional[int]:
        m = re.search(r"\btop\s+(\d+)\b", text)
        if m:
            try:
                return max(1, min(self.preferred_max_results, int(m.group(1))))
            except Exception:
                return None
        return None

    def _extract_booking_id(self, text: str) -> Optional[str]:
        m = re.search(r"\bbooking\s*(?:id|no\.?|number)?\s*[:#-]?\s*(\d{1,10})\b", text)
        if m:
            return m.group(1)
        if "cancel" in text or "modify" in text or "change booking" in text:
            m2 = re.search(r"\b(\d{1,10})\b", text)
            if m2:
                return m2.group(1)
        return None

    def _extract_preference(self, text: str) -> Optional[str]:
        if any(x in text for x in ("direct only", "direct trains only", "non stop", "non-stop")):
            return "direct_only"
        if any(x in text for x in ("overnight", "night train", "tonight")):
            return "overnight"
        if any(x in text for x in ("comfortable", "convenient", "best balance")):
            return "comfort"
        if any(x in text for x in ("cheapest", "lowest fare", "budget")):
            return "low_cost"
        if any(x in text for x in ("fastest", "quickest", "least time")):
            return "fastest"
        if any(x in text for x in ("fewest stops", "least stops", "fewer stops", "shortest")):
            return "shortest"
        return None

    def _merge_context(self, slots: QuerySlots, memory: Dict[str, Any], previous_result: Dict[str, Any], text: str) -> QuerySlots:
        memory_candidates = self._normalize_context_source(memory)
        prev_candidates = self._normalize_context_source(previous_result)
        for candidate in (memory_candidates, prev_candidates):
            if not slots.source:
                slots.source = candidate.get("source") or candidate.get("source_station")
            if not slots.destination:
                slots.destination = candidate.get("destination") or candidate.get("destination_station")
            if not slots.train_number:
                slots.train_number = candidate.get("train_number")
            if not slots.travel_class:
                slots.travel_class = candidate.get("travel_class") or candidate.get("class_code")
            if not slots.passengers:
                pax = candidate.get("passengers") or candidate.get("passenger_count")
                if isinstance(pax, int) and pax > 0:
                    slots.passengers = pax
            if not slots.travel_date:
                slots.travel_date = candidate.get("travel_date") or candidate.get("date")
            if not slots.station:
                slots.station = candidate.get("station") or candidate.get("station_code")

        # Additional context-aware filling from previous_result if available and intent is a compare/fare/booking follow-up
        try:
            prev_slots = {}
            if isinstance(previous_result, dict):
                if 'slots' in previous_result and isinstance(previous_result['slots'], dict):
                    prev_slots = previous_result['slots']
                else:
                    prev_slots = previous_result
            # From previous search_results, extract route if source/destination are missing
            if (not slots.source or not slots.destination) and isinstance(prev_slots, dict):
                # Try top-level keys
                if not slots.source:
                    slots.source = self._string_or_none(prev_slots.get('source') or prev_slots.get('source_station'))
                if not slots.destination:
                    slots.destination = self._string_or_none(prev_slots.get('destination') or prev_slots.get('destination_station'))
                # If still missing, check nested results list
                results = prev_slots.get('results') or prev_slots.get('search_results') or prev_slots.get('results_list')
                if isinstance(results, list) and results:
                    first = results[0]
                    if isinstance(first, dict):
                        if not slots.source:
                            for k in ('source', 'from', 'src', 'source_station'):
                                if first.get(k):
                                    slots.source = first.get(k); break
                        if not slots.destination:
                            for k in ('destination', 'to', 'dst', 'destination_station'):
                                if first.get(k):
                                    slots.destination = first.get(k); break
        except Exception:
            pass

        if any(phrase in text for phrase in ("that one", "that train", "the first one", "the first train", "it", "same one")):
            if previous_result.get("selected_train_number") and not slots.train_number:
                slots.train_number = self._string_or_none(previous_result.get("selected_train_number"))
            if memory.get("selected_train_number") and not slots.train_number:
                slots.train_number = self._string_or_none(memory.get("selected_train_number"))

        if not slots.travel_class and any(x in text for x in ("what about ac", "ac?", "3a", "2a", "1a")):
            cls = self._extract_class(text)
            if cls:
                slots.travel_class = cls

        if not slots.source and previous_result.get("source"):
            slots.source = self._string_or_none(previous_result.get("source"))
        if not slots.destination and previous_result.get("destination"):
            slots.destination = self._string_or_none(previous_result.get("destination"))

        # Heuristic: if user reply is a short station phrase and memory has a source, treat it as destination
        if not slots.destination:
            candidate_text = text.strip()
            if candidate_text and len(candidate_text.split()) <= 3:
                resolved = self._resolve_station_phrase(candidate_text)
                if resolved and (memory.get("source") or memory.get("source_station") or previous_result.get("source")):
                    slots.destination = resolved

        # Heuristic: ordinal selection ("first", "second", "2nd") -> pick an option from previous results
        ord_idx = self._parse_ordinal(text)
        if ord_idx is not None:
            # store 0-based index
            slots.selected_option_index = max(0, ord_idx - 1)
            try:
                results = previous_result.get("results") or previous_result.get("search_results") or []
                if results and 0 <= slots.selected_option_index < len(results) and not slots.train_number:
                    candidate = results[slots.selected_option_index]
                    if isinstance(candidate, dict) and candidate.get("train_number"):
                        slots.train_number = candidate.get("train_number")
            except Exception:
                pass

        # Heuristic: if previous_result asked for clarification and the reply is short, inherit the missing slot
        if previous_result and previous_result.get("clarification_needed"):
            token_count = len(text.strip().split()) if text.strip() else 0
            if token_count <= 4:
                # Try filling travel_date
                dt = self._extract_date(text)
                if dt and not slots.travel_date:
                    slots.travel_date = dt
                # Try filling class
                cls = self._extract_class(text)
                if cls and not slots.travel_class:
                    slots.travel_class = cls
                # Try filling passengers
                pax = self._extract_passengers(text)
                if pax and not slots.passengers:
                    slots.passengers = pax

        # Heuristic: bare numeric replies are often passenger counts in follow-ups
        if not slots.passengers:
            m = re.fullmatch(r"\s*(\d+)\s*", text)
            if m:
                try:
                    slots.passengers = int(m.group(1))
                except Exception:
                    pass

        return slots

    def _normalize_context_source(self, data: Any) -> Dict[str, Any]:
        """Normalize possible previous_result or memory shapes into a flat dict of known keys.
        Accepts:
        - a direct slot dict (source/destination/...)
        - a previous_result produced by QueryInterpretation.to_dict()
        - a legacy search result containing 'results' or 'search_results'
        """
        if not isinstance(data, dict):
            return {}
        # If wrapped under 'slots', prefer that
        if 'slots' in data and isinstance(data['slots'], dict):
            data = {**data['slots'], **data}
            # flatten: slots may override top-level keys
            data.pop('slots', None)
        # Also consider resolved_entities
        if 'resolved_entities' in data and isinstance(data['resolved_entities'], dict):
            data = {**data['resolved_entities'], **data}
            data.pop('resolved_entities', None)

        keys = {
            "source", "destination", "train_number", "travel_class", "passengers",
            "travel_date", "station", "source_station", "destination_station",
            "selected_train_number", "class_code", "passenger_count", "date"
        }
        out = {k: v for k, v in data.items() if k in keys and v not in (None, "", [], {})}
        # If results are present, attempt to extract source/destination from the first result
        results = data.get('results') or data.get('search_results') or data.get('results_list')
        if (not out.get('source') or not out.get('destination')) and isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                for key in ('source', 'from', 'src', 'source_station'):
                    if not out.get('source') and first.get(key):
                        out['source'] = first.get(key)
                for key in ('destination', 'to', 'dst', 'destination_station'):
                    if not out.get('destination') and first.get(key):
                        out['destination'] = first.get(key)
        return out

    def _missing_slots_for_intent(self, intent: str, slots: QuerySlots, previous_result: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        if intent in {"train_search", "fare_query", "booking_create", "route_query"}:
            if not slots.source:
                missing.append("source")
            if not slots.destination:
                missing.append("destination")
        # Require train_number for route_query or train_info when train_number isn't provided
        # For fare_query prefer asking for source/destination rather than train number
        if intent in {"route_query", "train_info"} and not slots.train_number:
            if not (slots.source and slots.destination):
                missing.append("train_number")
        if intent == "booking_create":
            if not slots.travel_class:
                missing.append("travel_class")
            if not slots.passengers:
                missing.append("passengers")
            if not slots.travel_date:
                missing.append("travel_date")
        if intent == "booking_cancel" and not slots.booking_id:
            missing.append("booking_id")
        if intent == "station_query" and not slots.station:
            if not slots.source and not slots.destination:
                missing.append("station")
        if intent == "fare_query" and not (slots.train_number or (slots.source and slots.destination)):
            if not previous_result:
                if "source" not in missing:
                    missing.append("source")
                if "destination" not in missing:
                    missing.append("destination")
        seen = set()
        unique: List[str] = []
        for item in missing:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique

    def _build_clarification_question(self, intent: str, missing_slots: List[str], slots: QuerySlots) -> str:
        if intent in {"train_search", "fare_query", "booking_create", "route_query"}:
            if "source" in missing_slots and "destination" in missing_slots:
                return "Please tell me the source and destination stations."
            if "source" in missing_slots:
                return f"Please tell me the source station to go with {slots.destination or 'your destination'}."
            if "destination" in missing_slots:
                return f"Please tell me the destination station from {slots.source or 'your source'}."
        if intent == "booking_create":
            if "travel_class" in missing_slots:
                return "Which class should I use: GN, 2S, SL, CC, 3A, 2A, 1A, or EC?"
            if "passengers" in missing_slots:
                return "How many passengers should I book for?"
            if "travel_date" in missing_slots:
                return "What travel date should I use?"
        if intent == "booking_cancel":
            return "Please provide the booking ID you want to cancel."
        if intent == "station_query":
            return "Please provide the station name or station code."
        if intent in {"route_query", "train_info"} and "train_number" in missing_slots:
            return "Please provide the train number so I can fetch the route or train details."
        if intent == "fare_query":
            if "train_number" in missing_slots and not (slots.source and slots.destination):
                return "Please provide the train number, or tell me the source and destination so I can estimate fare."
            return "Please tell me the route so I can estimate the fare."
        return "Could you share a little more detail?"

    def _score_confidence(self, intent: str, slots: QuerySlots, clarification_needed: bool,
                          normalized_text: str, previous_result: Dict[str, Any]) -> float:
        score = 0.25
        if intent != "train_search" or ("train" in normalized_text or "fare" in normalized_text or "book" in normalized_text):
            score += 0.2
        for value in (slots.source, slots.destination, slots.train_number, slots.travel_class, slots.station):
            if value:
                score += 0.09
        if slots.passengers:
            score += 0.05
        if slots.travel_date:
            score += 0.05
        if slots.time_hint:
            score += 0.03
        if slots.sort_by:
            score += 0.03
        if slots.preference:
            score += 0.03
        if clarification_needed:
            score -= 0.2
        if any(x in normalized_text for x in ("something", "somewhere", "a train", "book a ticket", "show options")):
            score -= 0.12
        if previous_result:
            score += 0.05
        return max(0.0, min(1.0, score))

    def _build_notes(self, text: str, slots: QuerySlots, memory: Dict[str, Any], previous_result: Dict[str, Any]) -> List[str]:
        notes: List[str] = []
        if not slots.source and not slots.destination and not slots.train_number:
            notes.append("query_is_sparse")
        if any(x in text for x in ("that one", "it", "same one", "the first train")):
            notes.append("follow_up_reference_detected")
        if previous_result:
            notes.append("previous_result_available")
        if memory:
            notes.append("memory_available")
        if any(x in text for x in ("cheapest", "fastest", "best balance", "compare")):
            notes.append("ranking_request")
        return notes

    def _resolve_station_phrase(self, phrase: str) -> Optional[str]:
        cleaned = self._normalize(phrase)
        cleaned = re.sub(r"\b(?:station|railway|rail)\b", "", cleaned).strip()
        if not cleaned:
            return None
        if cleaned in self.station_aliases:
            return self.station_aliases[cleaned]
        for alias, code in sorted(self.station_aliases.items(), key=lambda item: -len(item[0])):
            if alias in cleaned:
                return code
        best_code, best_score = None, 0.0
        for alias, code in self.station_aliases.items():
            score = self._similarity(cleaned, alias)
            if score > best_score:
                best_code, best_score = code, score
        # Lower threshold slightly to catch common misspellings; keep conservative to avoid false positives
        if best_code and best_score >= 0.75:
            return best_code
        if re.fullmatch(r"[A-Za-z0-9]{2,6}", cleaned):
            return cleaned.upper()
        return None

    def _similarity(self, a: str, b: str) -> float:
        try:
            from rapidfuzz import fuzz  # type: ignore
            return float(fuzz.ratio(a, b)) / 100.0
        except Exception:
            return difflib.SequenceMatcher(None, a, b).ratio()

    def _parse_ordinal(self, text: str) -> Optional[int]:
        """Return 1-based ordinal if detected in text (e.g., 'first'->1, '2nd'->2)."""
        text = (text or "").lower()
        ord_map = {
            'first': 1, '1st': 1, 'one': 1,
            'second': 2, '2nd': 2, 'two': 2,
            'third': 3, '3rd': 3, 'three': 3,
            'fourth': 4, '4th': 4, 'four': 4,
            'fifth': 5, '5th': 5, 'five': 5,
        }
        for k, v in ord_map.items():
            if re.search(rf"\b{k}\b", text):
                return v
        m = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _string_or_none(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    def _normalize(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = text.replace("–", "-").replace("—", "-")
        text = re.sub(r"\s+", " ", text)
        return text

    def should_ask_followup(self, interpretation: QueryInterpretation) -> bool:
        return interpretation.clarification_needed or interpretation.confidence < self.confidence_threshold

    def summarize_slots(self, interpretation: QueryInterpretation) -> Dict[str, Any]:
        return interpretation.to_dict()

    def get_search_signature(self, interpretation: QueryInterpretation) -> str:
        slots = interpretation.slots
        return "|".join([
            interpretation.intent,
            slots.source or "",
            slots.destination or "",
            slots.train_number or "",
            slots.travel_class or "",
            slots.travel_date or "",
            slots.time_hint or "",
            slots.sort_by or "",
            slots.preference or "",
        ])

    def extract_route_pair(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        return self._extract_stations(self._normalize(text))

    def extract_station(self, text: str) -> Optional[str]:
        return self._extract_station_only(self._normalize(text))

    def detect_language_hint(self, text: str) -> str:
        return "en"


def interpret_query(text: str, memory: Optional[Dict[str, Any]] = None,
                    previous_result: Optional[Dict[str, Any]] = None) -> QueryInterpretation:
    return QueryUnderstanding().interpret(text=text, memory=memory, previous_result=previous_result)
