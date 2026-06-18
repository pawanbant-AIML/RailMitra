"""backend/app/agent/query_understanding.py"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple
import difflib
import re


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
        "blr": "SBC",
        "sbc": "SBC",
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
        "majn": "MAJN",
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
        "new delhi": "NDLS",
        "ndls": "NDLS",
        "mumbai": "CSMT",
        "bombay": "CSMT",
        "kolkata": "HWH",
        "calcutta": "HWH",
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
        "ahmedabad": "ADI",
        "surat": "ST",
        "vadodara": "BRC",
        "baroda": "BRC",
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
        "varanasi": "BSB",
        "banaras": "BSB",
        "kashi": "BSB",
        "davangere": "DVG",
        "davanagere": "DVG",
        "shimoga": "SMET",
        "shivamogga": "SMET",
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
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    def __init__(
        self,
        station_aliases: Optional[Dict[str, str]] = None,
        class_aliases: Optional[Dict[str, str]] = None,
        confidence_threshold: float = 0.55,
        preferred_max_results: int = 10,
    ) -> None:
        self.station_aliases = {**self.STATION_ALIASES, **(station_aliases or {})}
        self.class_aliases = {**self.CLASS_ALIASES, **(class_aliases or {})}
        self.confidence_threshold = confidence_threshold
        self.preferred_max_results = preferred_max_results
        try:
            from rapidfuzz import fuzz  # type: ignore
            self._ratio = fuzz.ratio
        except Exception:  # pragma: no cover
            self._ratio = None

    def interpret(
        self,
        text: str,
        memory: Optional[Dict[str, Any]] = None,
        previous_result: Optional[Dict[str, Any]] = None,
    ) -> QueryInterpretation:
        raw_text = (text or "").strip()
        normalized_text = self._normalize(raw_text)
        memory = memory or {}
        previous_result = previous_result or {}

        intent = self._detect_intent(normalized_text, previous_result)
        sub_intents = self._detect_sub_intents(normalized_text)

        slots = QuerySlots(
            source=None,
            destination=None,
            train_number=None,
            travel_class=None,
            passengers=None,
            travel_date=None,
            time_hint=None,
            departure_after=None,
            departure_before=None,
            budget_max=None,
            sort_by=None,
            limit=None,
            booking_id=None,
            station=None,
            preference=None,
            selected_option_index=None,
        )
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
        missing_slots = self._missing_slots_for_intent(intent, slots)
        clarification_needed = bool(missing_slots) and intent in {
            "train_search", "fare_query", "booking_create", "booking_cancel",
            "booking_modify", "route_query", "train_info", "station_query",
            "multi_intent",
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
            "travel_date": slots.travel_date,
            "budget_max": slots.budget_max,
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

    def _normalize(self, text: str) -> str:
        text = (text or "").lower().strip()
        text = re.sub(r"[^\w\s₹/:-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _detect_intent(self, text: str, previous_result: Optional[Dict[str, Any]] = None) -> str:
        if previous_result and previous_result.get("clarification_needed"):
            if len(text.split()) <= 5 and previous_result.get("intent") in {"booking_create", "fare_query", "train_search", "route_query", "multi_intent"}:
                return previous_result["intent"]

        if self._extract_train_number(text):
            return "train_info"

        scored = {intent: 0 for intent in self.INTENT_PATTERNS}
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in text:
                    scored[intent] += 1

        if any(k in text for k in ("cheapest", "fastest", "compare", "best balance", "under ₹", "under rs", "sort by", "budget", "best option")):
            return "multi_intent"

        for intent in ("booking_cancel", "booking_modify", "booking_history", "booking_create"):
            if scored[intent]:
                return intent
        if scored["fare_query"]:
            return "fare_query"
        if scored["route_query"]:
            return "route_query"
        if scored["train_info"]:
            return "train_info"
        if scored["station_query"]:
            return "station_query"
        if scored["train_search"]:
            return "train_search"
        if scored["greeting"]:
            return "greeting"
        return "train_search"

    def _detect_sub_intents(self, text: str) -> List[str]:
        out: List[str] = []
        groups = [
            ("cheapest", ("cheapest", "lowest fare", "lowest price", "minimum cost")),
            ("fastest", ("fastest", "least time", "shortest journey", "quickest")),
            ("direct_only", ("direct only", "non stop", "non-stop", "without change")),
            ("fewest_stops", ("fewest stops", "least stops", "fewer stops")),
            ("compare", ("compare", "comparison", "vs", "versus")),
            ("overnight", ("overnight", "night train", "tonight")),
        ]
        for label, words in groups:
            if any(w in text for w in words):
                out.append(label)
        return out

    def _extract_stations(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        patterns = [
            r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+tomorrow|\s+today|\s+tonight|\s+after|\s+before|\s+at|\s+for|\s+in|\s*$)",
            r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+tomorrow|\s+today|\s+tonight|\s+after|\s+before|\s+at|\s+for|\s+in|\s*$)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return self._resolve_station_phrase(m.group(1)), self._resolve_station_phrase(m.group(2))

        found: List[Tuple[int, str]] = []
        padded = f" {text} "
        for alias, code in self.station_aliases.items():
            idx = padded.find(f" {alias} ")
            if idx != -1:
                found.append((idx, code))
        if found:
            found.sort(key=lambda x: x[0])
            codes = [c for _, c in found]
            if len(codes) >= 2:
                return codes[0], codes[1]
            return codes[0], None
        return None, None

    def _resolve_station_phrase(self, phrase: str) -> Optional[str]:
        phrase = self._clean_station_phrase(phrase)
        if not phrase:
            return None
        if phrase in self.station_aliases:
            return self.station_aliases[phrase]
        best_code, best_score = None, 0.0
        for alias, code in self.station_aliases.items():
            score = self._similarity(phrase, alias)
            if score > best_score:
                best_score = score
                best_code = code
        return best_code if best_score >= 80 else None

    def _clean_station_phrase(self, phrase: str) -> str:
        phrase = re.sub(r"\b(to|from|between|and|for|in|at|after|before|tomorrow|today|tonight)\b", " ", phrase, flags=re.I)
        phrase = re.sub(r"\s+", " ", phrase).strip().lower()
        return phrase

    def _extract_train_number(self, text: str) -> Optional[str]:
        budget_hit = bool(re.search(r"(under|below|less than|within|budget|max|upto|up to)\s*(?:rs\.?|inr|₹)?\s*\d+", text, re.I))
        patterns = [
            r"\btrain\s*(?:no\.?|number)?\s*[:#-]?\s*(\d{4,6})\b",
            r"\b(?:no\.?|number|route|details|about)\s*[:#-]?\s*(\d{4,6})\b",
            r"\b(\d{4,6})\b",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.I):
                num = m.group(1)
                window = text[max(0, m.start()-18): min(len(text), m.end()+18)]
                if re.search(r"(₹|rs\.?|inr|fare|cost|price|budget|under|below|less than|within|max|upto|up to)", window, re.I) and not re.search(r"\btrain\b", window, re.I):
                    continue
                if budget_hit and len(num) <= 4 and not re.search(r"\btrain\b", window, re.I):
                    continue
                if len(num) >= 5:
                    return num
                if re.search(r"\btrain\b", window, re.I):
                    return num
        return None

    def _extract_class(self, text: str) -> Optional[str]:
        normalized = text.lower()
        for alias, code in sorted(self.class_aliases.items(), key=lambda item: -len(item[0])):
            if alias in normalized:
                return code
        return None

    def _extract_passengers(self, text: str) -> Optional[int]:
        # Updated regex to allow an optional word between number and noun (e.g., "2 sleeper tickets")
        m = re.search(r"\b(\d+)\s*(?:\w+\s+)?(?:passenger|pax|ticket|seat|person|people|traveller|traveler)s?\b", text, re.I)
        if m:
            return int(m.group(1))
        for word, value in self.NUM_WORDS.items():
            if re.search(rf"\b{word}\b", text, re.I):
                return value
        return None

    def _extract_date(self, text: str) -> Optional[str]:
        today = date.today()
        if "day after tomorrow" in text:
            return (today + timedelta(days=2)).isoformat()
        if "tomorrow" in text:
            return (today + timedelta(days=1)).isoformat()
        if "today" in text:
            return today.isoformat()
        ymd = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
        if ymd:
            y, m, d = map(int, ymd.groups())
            try:
                return date(y, m, d).isoformat()
            except Exception:
                return None
        dmy = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", text)
        if dmy:
            d, m, y = map(int, dmy.groups())
            try:
                return date(y, m, d).isoformat()
            except Exception:
                return None
        return None

    def _extract_time_hint(self, text: str) -> Optional[str]:
        if any(w in text for w in ("morning", "early")):
            return "morning"
        if any(w in text for w in ("afternoon", "noon")):
            return "afternoon"
        if "evening" in text:
            return "evening"
        if any(w in text for w in ("night", "overnight", "tonight", "late")):
            return "night"
        m = re.search(r"\b(?:at|after|before)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.I)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2) or 0)
            ap = m.group(3).lower()
            if ap == "pm" and hh != 12:
                hh += 12
            if ap == "am" and hh == 12:
                hh = 0
            return f"{hh:02d}:{mm:02d}"
        return None

    def _extract_departure_after(self, text: str) -> Optional[str]:
        m = re.search(r"(?:after|from)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", text, re.I)
        return self._normalize_time_token(m.group(1)) if m else None

    def _extract_departure_before(self, text: str) -> Optional[str]:
        m = re.search(r"before\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", text, re.I)
        return self._normalize_time_token(m.group(1)) if m else None

    def _normalize_time_token(self, token: str) -> str:
        token = token.strip().lower().replace(" ", "")
        m = re.match(r"(\d{1,2})(?::(\d{2}))?(am|pm)", token)
        if not m:
            return token
        hh = int(m.group(1))
        mm = int(m.group(2) or 0)
        ap = m.group(3)
        if ap == "pm" and hh != 12:
            hh += 12
        if ap == "am" and hh == 12:
            hh = 0
        return f"{hh:02d}:{mm:02d}"

    def _extract_budget(self, text: str) -> Optional[int]:
        m = re.search(r"(?:under|less than|within|below|max|upto|up to|budget(?: of)?)\s*(?:Rs\.?|INR|₹)?\s*(\d+)", text, re.I)
        return int(m.group(1)) if m else None

    def _extract_sort_hint(self, text: str) -> Optional[str]:
        if any(x in text for x in ("cheapest", "lowest fare", "sort by fare", "budget")):
            return "fare"
        if any(x in text for x in ("fastest", "least time", "shortest journey", "sort by time")):
            return "duration"
        if any(x in text for x in ("fewest stops", "least stops", "fewer stops")):
            return "stops"
        return None

    def _extract_limit(self, text: str) -> Optional[int]:
        m = re.search(r"\btop\s+(\d+)\b", text, re.I)
        if m:
            return max(1, min(10, int(m.group(1))))
        return self.preferred_max_results

    def _extract_booking_id(self, text: str) -> Optional[str]:
        m = re.search(r"\b(?:pnr|booking\s*(?:id|no\.?|number)?)\s*[:#-]?\s*([a-z0-9]{4,12})\b", text, re.I)
        return m.group(1).upper() if m else None

    def _extract_station_only(self, text: str) -> Optional[str]:
        if re.search(r"(?:station|railway station|junction)", text, re.I):
            for alias, code in self.station_aliases.items():
                if re.search(rf"\b{re.escape(alias)}\b", text, re.I):
                    return code
        return None

    def _extract_preference(self, text: str) -> Optional[str]:
        if "direct only" in text or "non stop" in text or "non-stop" in text:
            return "direct_only"
        if "sleeper" in text:
            return "sleeper"
        if "ac" in text and not any(c in text for c in ("1ac", "2ac", "3ac", "1a", "2a", "3a")):
            return "ac"
        return None

    def _merge_context(
        self,
        slots: QuerySlots,
        memory: Dict[str, Any],
        previous_result: Optional[Dict[str, Any]],
        text: str,
    ) -> QuerySlots:
        mem_slots = memory.get("slots", {}) if isinstance(memory, dict) else {}
        for key in ("source", "destination", "travel_date", "travel_class", "passengers"):
            if getattr(slots, key) is None and mem_slots.get(key):
                setattr(slots, key, mem_slots[key])

        if previous_result:
            entities = previous_result.get("entities", {})
            for key, prev_key in (("source","source"), ("destination","destination"), ("travel_date","date"), ("travel_class","travel_class")):
                if getattr(slots, key) is None and entities.get(prev_key):
                    setattr(slots, key, entities[prev_key])
            if slots.passengers is None and entities.get("passengers"):
                try:
                    slots.passengers = int(entities["passengers"])
                except Exception:
                    pass
            ordinals = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}
            for word, idx in ordinals.items():
                if re.search(rf"\b{word}\b", text):
                    slots.selected_option_index = idx
                    break
        return slots

    def _missing_slots_for_intent(self, intent: str, slots: QuerySlots) -> List[str]:
        missing: List[str] = []
        if intent in {"train_search", "fare_query", "route_query", "booking_create"}:
            if not slots.source:
                missing.append("source")
            if not slots.destination:
                missing.append("destination")
        if intent == "booking_create":
            if not slots.travel_class:
                missing.append("travel_class")
            if not slots.passengers:
                missing.append("passengers")
            if not slots.travel_date:
                missing.append("travel_date")
        if intent == "booking_cancel" and not slots.booking_id:
            missing.append("booking_id")
        if intent == "train_info" and not slots.train_number:
            missing.append("train_number")
        if intent == "station_query" and not (slots.station or slots.source or slots.destination):
            missing.append("station")
        return missing

    def _build_clarification_question(self, intent: str, missing: List[str], slots: QuerySlots) -> str:
        if intent == "booking_create":
            parts = []
            mapping = {
                "source": "source station",
                "destination": "destination station",
                "travel_class": "class",
                "passengers": "number of passengers",
                "travel_date": "travel date",
            }
            for item in missing:
                if item in mapping:
                    parts.append(mapping[item])
            if parts:
                return "Please share " + ", ".join(parts) + " so I can book it."
        if intent in {"train_search", "fare_query", "route_query"}:
            if "source" in missing and "destination" in missing:
                return "Please tell me the source and destination stations."
            if "source" in missing:
                return "Please tell me the source station."
            if "destination" in missing:
                return "Please tell me the destination station."
        if intent == "booking_cancel":
            return "Please share the booking ID so I can cancel it."
        if intent == "train_info":
            return "Please provide the train number."
        if intent == "station_query":
            return "Please tell me which station you want to know about."
        return "Please share a little more detail so I can help."

    def _score_confidence(self, intent: str, slots: QuerySlots, clarification_needed: bool, text: str, previous_result: Optional[Dict[str, Any]]) -> float:
        score = 0.35
        if slots.source:
            score += 0.18
        if slots.destination:
            score += 0.18
        if slots.travel_date:
            score += 0.08
        if slots.travel_class:
            score += 0.08
        if slots.time_hint or slots.departure_after or slots.departure_before:
            score += 0.08
        if intent == "greeting":
            score = 0.9
        if clarification_needed:
            score -= 0.12
        if previous_result and previous_result.get("previous_results"):
            score += 0.08
        return max(0.05, min(0.99, score))

    def _build_notes(self, text: str, slots: QuerySlots, memory: Dict[str, Any], previous_result: Dict[str, Any]) -> List[str]:
        notes = []
        if slots.budget_max is not None:
            notes.append(f"budget<=₹{slots.budget_max}")
        if slots.time_hint:
            notes.append(f"time_hint={slots.time_hint}")
        if slots.departure_after:
            notes.append(f"after={slots.departure_after}")
        if slots.departure_before:
            notes.append(f"before={slots.departure_before}")
        if previous_result.get("previous_results"):
            notes.append("follow_up_context_available")
        if not notes:
            notes.append("no_special_notes")
        return notes

    def _similarity(self, a: str, b: str) -> float:
        if self._ratio is not None:
            return float(self._ratio(a, b))
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def build_query_understanding() -> QueryUnderstanding:
    return QueryUnderstanding()
