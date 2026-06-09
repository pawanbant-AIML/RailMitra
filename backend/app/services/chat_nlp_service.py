"""
NLP Service for train booking conversation analysis.

Drop-in replacement with:
- stronger intent detection
- better station/date/class/passenger extraction
- conversation memory from plain chat history
- follow-up clarification handling
- support for short replies like "tomorrow" after a date question
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from zoneinfo import ZoneInfo
from difflib import get_close_matches

import spacy
from pydantic import BaseModel


# ==================== SCHEMAS ====================

class Entity(BaseModel):
    source: Optional[str] = None
    destination: Optional[str] = None
    via_stations: Optional[List[str]] = None
    date: Optional[str] = None              # ISO-8601: YYYY-MM-DD
    time: Optional[str] = None              # HH:MM in 24-hour format
    travel_class: Optional[str] = None      # SL, 2A, 3A, 1A, CC, 2S, GN
    passengers: Optional[int] = None
    budget: Optional[float] = None
    preference: Optional[str] = None        # shortest, cheapest, fastest
    train_number: Optional[str] = None
    booking_id: Optional[int] = None


class ChatAnalysisRequest(BaseModel):
    user_message: str
    conversation_history: Optional[List[Dict[str, Any]]] = None   # Fix #1


class ChatAnalysisResponse(BaseModel):
    intent: str
    confidence: float
    entities: Entity
    resolved_context: Dict[str, Any]
    missing_required_slots: List[str]
    clarification_needed: bool
    clarification_question: Optional[str] = None
    next_action: str
    action_payload: Optional[Dict[str, Any]] = None
    memory_patch: Optional[Dict[str, Any]] = None


# ==================== STATION / CITY MAPPING ====================

CITY_TO_CODES = {
    "bangalore": ["SBC", "YPR"],
    "bengaluru": ["SBC", "YPR"],
    "blr": ["SBC"],
    "mumbai": ["CSTM", "BCT", "BDTS"],
    "bombay": ["CSTM", "BCT"],
    "mum": ["CSTM"],
    "delhi": ["NDLS", "DLI", "NZM"],
    "new delhi": ["NDLS"],
    "ndls": ["NDLS"],
    "dli": ["DLI"],
    "chennai": ["MAS", "MS"],
    "madras": ["MAS"],
    "chn": ["MAS"],
    "kolkata": ["HWH", "SDAH"],
    "calcutta": ["HWH"],
    "kol": ["HWH"],
    "hyderabad": ["SC", "HYB"],
    "secunderabad": ["SC"],
    "hyd": ["SC"],
    "pune": ["PUNE", "PNA"],
    "jaipur": ["JP", "JPI"],
    "lucknow": ["LKO", "LJN"],
    "patna": ["PNBE", "PPTA"],
    "bhopal": ["BPL"],
    "nagpur": ["NGP"],
    "surat": ["ST"],
    "ahmedabad": ["ADI"],
    "varanasi": ["BSB", "BSB"],
    "kochi": ["ERS"],
    "trivandrum": ["TVC"],
    "tvm": ["TVC"],
    "guwahati": ["GHY"],
    "amritsar": ["ASR"],
    "chandigarh": ["CDG"],
}

CLASS_ALIASES = {
    "second sitting": "2S",
    "sleeper": "SL",
    "sl": "SL",
    "general": "GN",
    "gn": "GN",
    "first ac": "1A",
    "1ac": "1A",
    "1a": "1A",
    "second ac": "2A",
    "2ac": "2A",
    "2a": "2A",
    "third ac": "3A",
    "3ac": "3A",
    "3a": "3A",
    "chair car": "CC",
    "cc": "CC",
    "2s": "2S",
    "ac": "2A",  # default AC
}

PREFERENCE_KEYWORDS = {
    "shortest route": "shortest",
    "minimum stops": "shortest",
    "fewest stops": "shortest",
    "cheapest route": "cheapest",
    "budget": "cheapest",
    "least fare": "cheapest",
    "fastest route": "fastest",
    "quickest": "fastest",
    "minimum time": "fastest",
}

# High‑priority intent phrases (used only for the new intent block)
BOOKING_HISTORY_WORDS = ["show my bookings", "my bookings"]
CANCEL_WORD = "cancel"
CHECK_ROUTE_WORDS = ["route for"]
ROUTE_SEARCH_WORDS = [
    "find train", "find trains", "search train", "search trains",
    "train from", "trains from"
]

FARE_PHRASES = (
    "fare from",
    "fare between",
    "how much",
    "cost from",
    "price from",
    "ticket price",
)

BOOK_PHRASES = (
    "book",
    "reserve",
    "ticket",
    "tickets",
    "booking",
)

COMPARE_PHRASES = (
    "compare",
    "which is better",
    "better route",
)

FASTEST_PHRASES = (
    "fastest",
    "quickest",
    "minimum time",
)

CHEAPEST_PHRASES = (
    "cheapest",
    "budget",
    "affordable",
    "low cost",
)

SHORTEST_PHRASES = (
    "shortest",
    "minimum stops",
    "fewest stops",
)

# Kept for backward compatibility if needed, but not used in new block
ROUTE_SEARCH_PHRASES = (
    "find trains",
    "search trains",
    "trains from",
    "train from",
    "between",
)


# ==================== NLP SERVICE ====================

class ChatNLPService:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Build a flat alias -> canonical code map
        self.station_alias_to_code: Dict[str, str] = {}
        for alias, codes in CITY_TO_CODES.items():
            if codes:
                self.station_alias_to_code[alias.lower()] = codes[0].upper()

        # Add direct station codes as aliases too
        for codes in CITY_TO_CODES.values():
            for code in codes:
                self.station_alias_to_code[code.lower()] = code.upper()

    # -------------------- Public API --------------------

    def analyze(self, request: ChatAnalysisRequest) -> ChatAnalysisResponse:
        """Main entry point for chat analysis."""
        user_message = (request.user_message or "").strip()
        history = request.conversation_history or []

        # Build context from conversation history first
        memory = self._build_memory(history)

        # Extract current message entities using previous context as a hint
        current_entities = self._extract_entities(user_message, memory)

        # Merge memory + current entities into one context
        resolved_context = self._merge_context(memory, current_entities.model_dump(exclude_none=True))
        public_context = self._public_context(resolved_context)

        # Detect intent, using context when the user reply is short
        intent = self._detect_intent(user_message, public_context, memory)

        # If the user is answering a pending clarification, preserve that flow
        current_slot_filled = self._filled_slots(current_entities)
        pending_slot = memory.get("_pending_slot")

        # Heuristic: short replies should continue the previous flow when possible
        if intent == "UNKNOWN":
            intent = self._infer_intent_from_context(user_message, public_context, memory, current_entities)

        missing_slots = self._check_missing_slots(intent, public_context)

        # If the last assistant asked for a specific slot and the user has not filled it,
        # keep asking that same question instead of dropping back to a generic response.
        if pending_slot and pending_slot not in current_slot_filled:
            if pending_slot not in missing_slots:
                missing_slots = [pending_slot] + missing_slots

        clarification_needed = len(missing_slots) > 0
        clarification_question = self._generate_clarification(missing_slots) if clarification_needed else None

        next_action = self._determine_next_action(intent, clarification_needed)
        action_payload = self._build_action_payload(intent, public_context, next_action)
        confidence = self._calculate_confidence(intent, current_entities, missing_slots)

        return ChatAnalysisResponse(
            intent=intent,
            confidence=confidence,
            entities=current_entities,
            resolved_context=public_context,
            missing_required_slots=missing_slots,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            next_action=next_action,
            action_payload=action_payload,
            memory_patch=current_entities.model_dump(exclude_none=True),
        )

    # -------------------- Memory --------------------

    # Fix #2: NEW simple memory builder that scans all messages for entities
    def _build_memory(self, history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Recover source/destination/date/class/passengers from previous conversation."""
        memory = {}

        if not history:
            return memory

        for msg in history:
            text = str(msg.get("content", "")).lower()

            source, destination = self._extract_stations(text, memory)  # context not needed here
            if source:
                memory["source"] = source
            if destination:
                memory["destination"] = destination

            date = self._extract_date(text)
            if date:
                memory["date"] = date

            travel_class = self._extract_class(text)
            if travel_class:
                memory["travel_class"] = travel_class

            passengers = self._extract_passengers(text)
            if passengers:
                memory["passengers"] = passengers

        return memory

    def _merge_context(self, base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        """Merge patch into base, preferring patch values for explicit new information."""
        merged = dict(base)
        for key, value in patch.items():
            if value is not None:
                merged[key] = value
        return merged

    def _public_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Strip internal keys before returning context to the client."""
        return {k: v for k, v in context.items() if not k.startswith("_")}

    # -------------------- Intent Detection --------------------

    def _detect_intent(self, message: str, context: Dict[str, Any], memory: Dict[str, Any]) -> str:
        msg = (message or "").lower().strip()

        if not msg:
            return "UNKNOWN"

        # ---------- Fix #3: stronger high-priority intents ----------
        if any(word in msg for word in BOOKING_HISTORY_WORDS):
            return "BOOKING_HISTORY"

        if CANCEL_WORD in msg:
            return "CANCEL_BOOKING"

        if any(word in msg for word in CHECK_ROUTE_WORDS):
            return "CHECK_ROUTE"

        if any(word in msg for word in ROUTE_SEARCH_WORDS):
            return "ROUTE_SEARCH"

        # ---------- keep the rest of the old intent logic ----------
        if any(p in msg for p in FARE_PHRASES):
            return "FARE_ESTIMATE"

        if any(p in msg for p in COMPARE_PHRASES):
            return "COMPARE_ROUTES"

        if any(p in msg for p in BOOK_PHRASES) and ("book" in msg or "reserve" in msg):
            return "BOOK_TICKET"

        if any(p in msg for p in FASTEST_PHRASES):
            return "FASTEST_ROUTE"

        if any(p in msg for p in CHEAPEST_PHRASES):
            return "CHEAPEST_ROUTE"

        if any(p in msg for p in SHORTEST_PHRASES):
            return "SHORTEST_ROUTE"

        # Fallback: if the message still looks like a station query, treat as ROUTE_SEARCH
        if (" from " in msg or " to " in msg or " between " in msg) and self._find_station_mentions(msg):
            return "ROUTE_SEARCH"

        return "UNKNOWN"

    def _infer_intent_from_context(
        self,
        message: str,
        context: Dict[str, Any],
        memory: Dict[str, Any],
        current_entities: Entity,
    ) -> str:
        """
        When the user gives a short follow-up reply, infer the active flow
        from the available context.
        """
        if context.get("source") and context.get("destination"):
            # Booking-like flow if we already have booking-related pieces
            booking_related = any(
                [
                    context.get("travel_class"),
                    context.get("passengers"),
                    current_entities.travel_class,
                    current_entities.passengers,
                    memory.get("_pending_slot") in {"travel_class", "passengers"},
                    "book" in (message or "").lower(),
                    "reserve" in (message or "").lower(),
                ]
            )
            if booking_related:
                return "BOOK_TICKET"

            # Otherwise, if the user has route fields or is replying to a date question,
            # assume route search.
            if any(
                [
                    context.get("date"),
                    current_entities.date,
                    memory.get("_pending_slot") == "date",
                    "tomorrow" in (message or "").lower(),
                    "today" in (message or "").lower(),
                    "next " in (message or "").lower(),
                ]
            ):
                return "ROUTE_SEARCH"

        return "UNKNOWN"

    # -------------------- Entity Extraction --------------------

    def _extract_entities(self, message: str, context: Dict[str, Any]) -> Entity:
        """Extract entities from the current message, remembering previous context."""
        msg = (message or "").lower()
        entities = Entity()

        # ---------- Fix #4: pre-populate from memory (context) ----------
        entities.source = context.get("source")
        entities.destination = context.get("destination")
        entities.date = context.get("date")
        entities.travel_class = context.get("travel_class")
        entities.passengers = context.get("passengers")

        # Now parse the current message – new values will override the defaults
        source, destination = self._extract_stations(msg, context)
        if source:
            entities.source = source
        if destination:
            entities.destination = destination

        via = self._extract_via_stations(msg)
        if via:
            entities.via_stations = via

        date = self._extract_date(msg)
        if date:
            entities.date = date

        time = self._extract_time(msg)
        if time:
            entities.time = time

        travel_class = self._extract_class(msg)
        if travel_class:
            entities.travel_class = travel_class

        passengers = self._extract_passengers(msg)
        if passengers is not None:
            entities.passengers = passengers

        budget = self._extract_budget(msg)
        if budget is not None:
            entities.budget = budget

        preference = self._extract_preference(msg)
        if preference:
            entities.preference = preference

        booking_id = self._extract_booking_id(msg)
        if booking_id is not None:
            entities.booking_id = booking_id

        train_number = self._extract_train_number(msg)
        if train_number:
            entities.train_number = train_number

        return entities

    def _find_station_mentions(self, message: str) -> List[Tuple[int, int, str]]:
        """
        Find all station/city aliases inside the message.
        Returns list of (start, end, canonical_code).
        """
        mentions: List[Tuple[int, int, str]] = []

        # Sort aliases by length so "new delhi" is checked before "delhi"
        aliases = sorted(self.station_alias_to_code.items(), key=lambda item: len(item[0]), reverse=True)

        for alias, code in aliases:
            pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
            for match in re.finditer(pattern, message):
                mentions.append((match.start(), match.end(), code))

        # Remove overlapping shorter matches
        mentions.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        filtered: List[Tuple[int, int, str]] = []
        last_end = -1
        for start, end, code in mentions:
            if start < last_end:
                continue
            filtered.append((start, end, code))
            last_end = end

        return filtered

    def _station_from_segment(self, segment: str) -> Optional[str]:
        seg = (segment or "").lower().strip()
        if not seg:
            return None

        mentions = self._find_station_mentions(seg)
        if mentions:
            return mentions[0][2]

        # Direct station-code fallback
        token = re.sub(r"[^a-z0-9]", "", seg)
        if token and len(token) <= 5 and token.isalpha():
            return token.upper()

        # Close match against known cities
        city_matches = get_close_matches(seg, self.station_alias_to_code.keys(), n=1, cutoff=0.72)
        if city_matches:
            return self.station_alias_to_code[city_matches[0]]

        return None

    def _extract_stations(self, message: str, context: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract source and destination in a way that works for:
        - "from Bangalore to Mumbai"
        - "Bangalore to Mumbai"
        - "between Pune and Hyderabad"
        - short follow-ups like "mumbai" (won't overwrite already-known source/destination)
        """
        source = None
        destination = None

        msg = message.lower()
        mentions = self._find_station_mentions(msg)

        # between X and Y
        between_match = re.search(
            r"between\s+(.+?)\s+and\s+(.+?)(?:\s+via|\s+on|\s+for|\s+at|\s+tomorrow|\s+today|$)",
            msg,
        )
        if between_match:
            source = self._station_from_segment(between_match.group(1))
            destination = self._station_from_segment(between_match.group(2))

        # from X to Y
        if source is None and destination is None and " from " in msg and " to " in msg:
            from_to_match = re.search(
                r"from\s+(.+?)\s+to\s+(.+?)(?:\s+via|\s+on|\s+for|\s+at|\s+tomorrow|\s+today|$)",
                msg,
            )
            if from_to_match:
                source = self._station_from_segment(from_to_match.group(1))
                destination = self._station_from_segment(from_to_match.group(2))

        # X to Y
        if (source is None or destination is None) and " to " in msg:
            left, right = msg.split(" to ", 1)
            left_code = self._station_from_segment(left)
            right_code = self._station_from_segment(right)

            if source is None and left_code:
                source = left_code
            if destination is None and right_code:
                destination = right_code

        # If we only found one station mention, use context to decide whether it's source or destination
        if source is None and destination is None and len(mentions) == 1:
            only_code = mentions[0][2]

            if context.get("source") and not context.get("destination"):
                destination = only_code
            elif context.get("destination") and not context.get("source"):
                source = only_code
            elif context.get("_pending_slot") == "source":
                source = only_code
            elif context.get("_pending_slot") == "destination":
                destination = only_code
            else:
                # If both source and destination are already known, do not overwrite on a stray single-city reply.
                pass

        # If we got two mentions without clear syntax, use the first two
        if source is None and destination is None and len(mentions) >= 2:
            source = mentions[0][2]
            destination = mentions[1][2]

        return source, destination

    def _extract_via_stations(self, message: str) -> Optional[List[str]]:
        via_stations: List[str] = []

        match = re.search(r"via\s+([^,]+?)(?:\s+to|\s+arrive|\s+on|\s+for|\s+at|$)", message)
        if match:
            via_text = match.group(1).strip()
            parts = [p.strip() for p in via_text.split(",") if p.strip()]
            for part in parts:
                station = self._station_from_segment(part)
                if station:
                    via_stations.append(station)

        return via_stations or None

    def _extract_date(self, message: str) -> Optional[str]:
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        msg = message.lower()

        if "today" in msg:
            return today.isoformat()

        if "tomorrow" in msg:
            return (today + timedelta(days=1)).isoformat()

        weekday_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for day_name in weekday_names:
            if f"next {day_name}" in msg:
                target_idx = weekday_names.index(day_name)
                delta = (target_idx - today.weekday()) % 7
                if delta == 0:
                    delta = 7
                return (today + timedelta(days=delta)).isoformat()

        # YYYY-MM-DD
        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", msg)
        if match:
            return match.group(1)

        # DD/MM/YYYY or DD-MM-YYYY or DD/MM or DD-MM
        match = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", msg)
        if match:
            day = int(match.group(1))
            month = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else today.year

            # Normalize 2-digit year
            if year < 100:
                year += 2000

            try:
                date_obj = datetime(year, month, day).date()
                if not match.group(3) and date_obj < today:
                    # If month/day without year already passed, assume next year
                    date_obj = datetime(year + 1, month, day).date()
                return date_obj.isoformat()
            except ValueError:
                pass

        # "10 june", "10 june 2026"
        match = re.search(
            r"\b(\d{1,2})\s+"
            r"(january|february|march|april|may|june|july|august|september|october|november|december)"
            r"(?:\s+(\d{4}))?\b",
            msg,
        )
        if match:
            day = int(match.group(1))
            month_name = match.group(2)
            year = int(match.group(3)) if match.group(3) else today.year
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12,
            }
            month = month_map[month_name]
            try:
                date_obj = datetime(year, month, day).date()
                if not match.group(3) and date_obj < today:
                    date_obj = datetime(year + 1, month, day).date()
                return date_obj.isoformat()
            except ValueError:
                pass

        return None

    def _extract_time(self, message: str) -> Optional[str]:
        msg = message.lower()

        match = re.search(r"\b(\d{1,2}):(\d{2})\b", msg)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            if 0 <= hour < 24 and 0 <= minute < 60:
                return f"{hour:02d}:{minute:02d}"

        match = re.search(r"\b(\d{1,2})(?:\s?)(am|pm)\b", msg)
        if match:
            hour = int(match.group(1))
            meridiem = match.group(2)
            if 1 <= hour <= 12:
                if meridiem == "pm" and hour != 12:
                    hour += 12
                if meridiem == "am" and hour == 12:
                    hour = 0
                return f"{hour:02d}:00"

        if "morning" in msg:
            return "08:00"
        if "afternoon" in msg:
            return "14:00"
        if "evening" in msg:
            return "18:00"
        if "night" in msg:
            return "21:00"

        return None

    def _extract_class(self, message: str) -> Optional[str]:
        msg = message.lower()

        # Prefer longer aliases first, so "second sitting" wins over "second"
        for alias in sorted(CLASS_ALIASES.keys(), key=len, reverse=True):
            if alias in msg:
                return CLASS_ALIASES[alias]

        return None

    def _extract_passengers(self, message: str) -> Optional[int]:
        msg = message.lower()

        patterns = [
            r"\b(?:book|reserve)\s+(\d{1,2})\s+(?:ticket|tickets|seat|seats|passenger|passengers|person|people)\b",
            r"\b(\d{1,2})\s+(?:ticket|tickets|seat|seats|passenger|passengers|person|people)\b",
            r"\b(\d{1,2})\s+(?:sleeper|sl|2a|3a|1a|cc|2s|ac)\s*(?:ticket|tickets|seat|seats)?\b",
            r"\bfor\s+(\d{1,2})\s+(?:passengers?|people|persons?|tickets?)\b",
            r"\b(\d{1,2})\s+(?:adults?|children|kids?|seniors?)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, msg)
            if match:
                n = int(match.group(1))
                if 1 <= n <= 20:
                    return n

        return None

    def _extract_budget(self, message: str) -> Optional[float]:
        msg = message.lower()

        match = re.search(r"[₹$]\s*(\d+(?:,\d{3})*(?:\.\d{2})?)", msg)
        if match:
            return float(match.group(1).replace(",", ""))

        match = re.search(r"\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(rupees|inr|dollars|usd)\b", msg)
        if match:
            return float(match.group(1).replace(",", ""))

        return None

    def _extract_preference(self, message: str) -> Optional[str]:
        msg = message.lower()

        for keyword in sorted(PREFERENCE_KEYWORDS.keys(), key=len, reverse=True):
            if keyword in msg:
                return PREFERENCE_KEYWORDS[keyword]

        return None

    def _extract_booking_id(self, message: str) -> Optional[int]:
        msg = message.lower()

        patterns = [
            r"\bbooking\s*#?\s*(\d+)\b",
            r"\bcancel(?:\s+booking)?\s*#?\s*(\d+)\b",
            r"\bbooking\s+id\s*#?\s*(\d+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, msg)
            if match:
                return int(match.group(1))

        return None

    def _extract_train_number(self, message: str) -> Optional[str]:
        msg = message.lower()

        match = re.search(r"\b(?:train|train number)\s*#?\s*(\d{3,6})\b", msg)
        if match:
            return match.group(1)

        # only accept a bare number if route/train context is present
        if any(k in msg for k in ["train", "route", "station"]):
            match = re.search(r"\b(\d{3,6})\b", msg)
            if match:
                return match.group(1)

        return None

    def _filled_slots(self, entities: Entity) -> List[str]:
        filled: List[str] = []
        data = entities.model_dump(exclude_none=True)
        for key in ["source", "destination", "date", "travel_class", "passengers", "train_number", "booking_id"]:
            if key in data:
                filled.append(key)
        return filled

    # -------------------- Clarification / Missing Slots --------------------

    def _check_missing_slots(self, intent: str, context: Dict[str, Any]) -> List[str]:
        missing: List[str] = []

        if intent in ["ROUTE_SEARCH", "SHORTEST_ROUTE", "CHEAPEST_ROUTE", "FASTEST_ROUTE", "COMPARE_ROUTES"]:
            if not context.get("source"):
                missing.append("source")
            if not context.get("destination"):
                missing.append("destination")
            if not context.get("date"):
                missing.append("date")

        elif intent == "BOOK_TICKET":
            if not context.get("source"):
                missing.append("source")
            if not context.get("destination"):
                missing.append("destination")
            if not context.get("date"):
                missing.append("date")
            if not context.get("travel_class"):
                missing.append("travel_class")
            if not context.get("passengers"):
                missing.append("passengers")

        elif intent == "FARE_ESTIMATE":
            if not context.get("source"):
                missing.append("source")
            if not context.get("destination"):
                missing.append("destination")

        elif intent == "CANCEL_BOOKING":               # <-- renamed from CANCEL_TICKET
            if not context.get("booking_id"):
                missing.append("booking_id")

        elif intent == "CHECK_ROUTE":
            if not context.get("train_number"):
                missing.append("train_number")

        return missing

    def _guess_clarification_slot(self, assistant_text: str) -> Optional[str]:
        """
        Infer what the assistant asked for from the text so we can keep context.
        """
        text = (assistant_text or "").lower()

        if "where are you traveling from" in text or "source" in text:
            return "source"
        if "where are you traveling to" in text or "destination" in text:
            return "destination"
        if "when do you want to travel" in text or "travel date" in text:
            return "date"
        if "what class" in text or "which class" in text or "preferred class" in text:
            return "travel_class"
        if "how many passengers" in text or "how many tickets" in text:
            return "passengers"
        if "what's your budget" in text or "your budget" in text:
            return "budget"
        if "booking id" in text:
            return "booking_id"
        if "train number" in text:
            return "train_number"

        return None

    def _generate_clarification(self, missing_slots: List[str]) -> Optional[str]:
        if not missing_slots:
            return None

        slot = missing_slots[0]
        questions = {
            "source": "Where are you traveling from?",
            "destination": "Where are you traveling to?",
            "date": "When do you want to travel?",
            "travel_class": "What class would you prefer (Sleeper, 2AC, 3AC, 1AC, CC, 2S)?",
            "passengers": "How many passengers?",
            "budget": "What's your budget?",
            "booking_id": "Please provide your Booking ID.",
            "train_number": "Please provide the train number.",
        }
        return questions.get(slot, f"Please provide {slot}.")

    # -------------------- Next Action / Payload --------------------

    # Fix #5: update and add entries
    def _determine_next_action(self, intent: str, clarification_needed: bool) -> str:
        if clarification_needed:
            return "ASK_CLARIFICATION"

        if intent == "ROUTE_SEARCH":
            return "SEARCH_ROUTE"
        if intent in ["SHORTEST_ROUTE", "CHEAPEST_ROUTE", "FASTEST_ROUTE"]:
            return "ROUTE_ANALYSIS"
        if intent == "COMPARE_ROUTES":
            return "COMPARE_ROUTES"
        if intent == "FARE_ESTIMATE":
            return "ESTIMATE_FARE"
        if intent == "BOOK_TICKET":
            return "BOOK"
        if intent == "BOOKING_HISTORY":
            return "BOOKING_HISTORY"
        if intent == "CANCEL_BOOKING":              # changed from CANCEL_TICKET
            return "CANCEL_BOOKING"
        if intent == "CHECK_ROUTE":
            return "CHECK_ROUTE"

        return "UNKNOWN"

    def _build_action_payload(self, intent: str, context: Dict[str, Any], next_action: str) -> Optional[Dict[str, Any]]:
        if next_action in ["ASK_CLARIFICATION", "UNKNOWN"]:
            return None

        if next_action in ["SEARCH_ROUTE", "ROUTE_ANALYSIS", "ESTIMATE_FARE", "COMPARE_ROUTES"]:
            return {
                "source": context.get("source"),
                "destination": context.get("destination"),
                "date": context.get("date"),
                "via_stations": context.get("via_stations"),
                "travel_class": context.get("travel_class"),
                "passengers": context.get("passengers", 1),
                "preference": context.get("preference"),
            }

        if next_action == "BOOK":
            return {
                "source": context.get("source"),
                "destination": context.get("destination"),
                "date": context.get("date"),
                "travel_class": context.get("travel_class"),
                "passengers": context.get("passengers"),
            }

        if next_action == "BOOKING_HISTORY":
            return {}

        if next_action == "CANCEL_BOOKING":         # renamed
            return {
                "booking_id": context.get("booking_id"),
            }

        if next_action == "CHECK_ROUTE":
            return {
                "train_number": context.get("train_number"),
            }

        return None

    def _calculate_confidence(self, intent: str, entities: Entity, missing_slots: List[str]) -> float:
        confidence = 0.45

        if intent in ["BOOK_TICKET", "ROUTE_SEARCH", "FARE_ESTIMATE", "BOOKING_HISTORY", "CANCEL_BOOKING", "CHECK_ROUTE"]:
            confidence = 0.8
        elif intent == "UNKNOWN":
            confidence = 0.2

        confidence -= min(len(missing_slots) * 0.12, 0.35)

        entity_count = sum(1 for v in entities.model_dump(exclude_none=True).values() if v is not None)
        confidence += min(entity_count * 0.05, 0.2)

        return max(0.1, min(1.0, confidence))
