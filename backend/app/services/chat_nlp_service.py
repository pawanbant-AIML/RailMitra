"""
Flexible Indian Railways conversational NLP service.
Handles typos, abbreviations, corrections, and casual English chat.
No Hinglish normalisation – only standard English processing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import spacy
from pydantic import BaseModel
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span

# =============================================================================
# Schemas
# =============================================================================


class Entity(BaseModel):
    source: Optional[str] = None
    destination: Optional[str] = None
    via_stations: Optional[List[str]] = None
    date: Optional[str] = None
    time: Optional[str] = None
    travel_class: Optional[str] = None
    passengers: Optional[int] = None
    budget: Optional[float] = None
    preference: Optional[str] = None
    train_number: Optional[str] = None
    booking_id: Optional[int] = None


class ChatAnalysisRequest(BaseModel):
    user_message: str
    conversation_history: Optional[List[Dict[str, Any]]] = None


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


# =============================================================================
# Station dictionary / aliases (exhaustive but extendable)
# =============================================================================


STATION_ALIASES: Dict[str, str] = {
    "bangalore": "SBC", "bengaluru": "SBC", "blr": "SBC",
    "bangalore city": "SBC", "ksr bengaluru": "SBC", "sbc": "SBC",
    "yesvantpur": "YPR", "ypr": "YPR", "bangalore cantonment": "BAND", "band": "BAND",
    "mysore": "MYS", "mysuru": "MYS", "mys": "MYS",
    "hubli": "UBL", "hubballi": "UBL", "ubl": "UBL",
    "mangalore": "MAQ", "mangaluru": "MAQ", "maq": "MAQ",
    "mumbai": "CSMT", "bombay": "CSMT", "bom": "CSMT", "csmt": "CSMT",
    "cstm": "CSMT", "vt": "CSMT", "mumbai central": "BCT", "bct": "BCT",
    "lokmanya tilak terminus": "LTT", "ltt": "LTT", "dadar": "DR", "dr": "DR",
    "bandra terminus": "BDTS", "bdts": "BDTS", "pune": "PUNE", "puna": "PUNE",
    "nashik": "NK", "nasik": "NK", "nk": "NK", "nagpur": "NGP", "ngp": "NGP",
    "aurangabad": "AWB", "awb": "AWB", "kolhapur": "KOP", "kop": "KOP",
    "solapur": "SUR", "sur": "SUR", "akola": "AK", "ak": "AK", "amravati": "AMI",
    "delhi": "NDLS", "new delhi": "NDLS", "ndls": "NDLS", "old delhi": "DLI",
    "dli": "DLI", "hazrat nizamuddin": "NZM", "nzm": "NZM", "anand vihar": "ANVT",
    "anvt": "ANVT", "jaipur": "JP", "jp": "JP", "jodhpur": "JU", "ju": "JU",
    "udaipur": "UDZ", "udz": "UDZ", "kota": "KOTA", "bikaner": "BKN", "bkn": "BKN",
    "ajmer": "AII", "aii": "AII", "agra": "AGC", "agra cantt": "AGC", "agc": "AGC",
    "amritsar": "ASR", "asr": "ASR", "chandigarh": "CDG", "cdg": "CDG",
    "ludhiana": "LDH", "ldh": "LDH", "bathinda": "BTI", "bti": "BTI",
    "chennai": "MAS", "madras": "MAS", "mas": "MAS", "chennai central": "MAS",
    "chennai egmore": "MS", "ms": "MS", "coimbatore": "CBE", "cbe": "CBE",
    "madurai": "MDU", "mdu": "MDU", "tiruchirapalli": "TPJ", "trichy": "TPJ",
    "tpj": "TPJ", "tirunelveli": "TEN", "ten": "TEN", "puducherry": "PDY",
    "pondicherry": "PDY", "pdy": "PDY", "salem": "SA", "sa": "SA",
    "erode": "ED", "ed": "ED", "tirupati": "TPTY", "tpty": "TPTY",
    "chengalpattu": "CGL", "vellore": "KPD",
    "kolkata": "HWH", "calcutta": "HWH", "howrah": "HWH", "hwh": "HWH",
    "sealdah": "SDAH", "sdah": "SDAH", "asansol": "ASN", "asn": "ASN",
    "dhanbad": "DHN", "dhn": "DHN", "ranchi": "RNC", "rnc": "RNC",
    "jamshedpur": "TATA", "tatanagar": "TATA", "bhubaneswar": "BBS", "bbs": "BBS",
    "puri": "PURI", "cuttack": "CTC", "ctc": "CTC", "new jalpaiguri": "NJP",
    "njp": "NJP", "siliguri": "SGUJ", "guwahati": "GHY", "gauhati": "GHY", "ghy": "GHY",
    "hyderabad": "HYB", "hyd": "HYB", "hyb": "HYB", "secunderabad": "SC", "sc": "SC",
    "kachiguda": "KCG", "vijayawada": "BZA", "bza": "BZA", "visakhapatnam": "VSKP",
    "vizag": "VSKP", "vskp": "VSKP", "guntur": "GNT", "gnt": "GNT",
    "rajahmundry": "RJY", "rajamahendravaram": "RJY", "rjy": "RJY",
    "kakinada": "CCT", "cct": "CCT", "tirupati": "TPTY",
    "ahmedabad": "ADI", "amdavad": "ADI", "adi": "ADI", "vadodara": "BRC",
    "baroda": "BRC", "brc": "BRC", "surat": "ST", "st": "ST", "rajkot": "RJT",
    "rjt": "RJT", "bhavnagar": "BVC", "bvc": "BVC", "gandhinagar": "GNC",
    "lucknow": "LKO", "lko": "LKO", "varanasi": "BSB", "banaras": "BSB",
    "kashi": "BSB", "bsb": "BSB", "patna": "PNBE", "pnbe": "PNBE",
    "prayagraj": "PRYJ", "allahabad": "PRYJ", "pryj": "PRYJ",
    "gorakhpur": "GKP", "gkp": "GKP", "gaya": "GAYA", "muzaffarpur": "MFP",
    "mfp": "MFP", "kanpur": "CNB", "cawnpore": "CNB", "cnb": "CNB",
    "mathura": "MTJ", "mtj": "MTJ", "meerut": "MTC", "bareilly": "BE",
    "be": "BE", "moradabad": "MB",
    "bhopal": "BPL", "bpl": "BPL", "gwalior": "GWL", "gwl": "GWL",
    "jabalpur": "JBP", "jbp": "JBP", "indore": "INDB", "indb": "INDB",
    "raipur": "R", "bilaspur": "BSP", "bsp": "BSP",
    "kochi": "ERS", "cochin": "ERS", "ernakulam": "ERS", "ers": "ERS",
    "thiruvananthapuram": "TVC", "trivandrum": "TVC", "tvc": "TVC",
    "kannur": "CAN", "cannanore": "CAN", "kozhikode": "CLT", "calicut": "CLT",
    "clt": "CLT", "thrissur": "TCR", "trichur": "TCR", "tcr": "TCR",
    "palakkad": "PGT", "palghat": "PGT", "pgt": "PGT", "alleppey": "ALLP",
    "alappuzha": "ALLP", "allp": "ALLP", "kollam": "QLN", "quilon": "QLN", "qln": "QLN",
}

CLASS_ALIASES: Dict[str, str] = {
    "executive chair car": "EC", "executive class": "EC", "ac first class": "1A",
    "ac 1st class": "1A", "first ac": "1A", "1st ac": "1A", "ac first": "1A",
    "second ac": "2A", "2nd ac": "2A", "ac 2nd class": "2A", "2 tier ac": "2A",
    "two tier ac": "2A", "ac 2 tier": "2A", "third ac": "3A", "3rd ac": "3A",
    "ac 3rd class": "3A", "3 tier ac": "3A", "three tier ac": "3A",
    "ac three tier": "3A", "ac chair car": "CC", "chair car": "CC",
    "second sitting": "2S", "sleeper class": "SL", "sleeper": "SL",
    "general": "GN", "unreserved": "GN", "ec": "EC", "1ac": "1A", "1a": "1A",
    "2ac": "2A", "2a": "2A", "3ac": "3A", "3a": "3A", "cc": "CC",
    "2s": "2S", "sl": "SL", "gn": "GN", "ac": "2A",
}

PREFERENCE_PHRASES: Dict[str, str] = {
    "shortest route": "shortest", "minimum stops": "shortest",
    "fewest stops": "shortest", "least stops": "shortest", "shortest path": "shortest",
    "cheapest route": "cheapest", "cheapest train": "cheapest",
    "least fare": "cheapest", "budget friendly": "cheapest",
    "low cost": "cheapest", "most affordable": "cheapest",
    "fastest route": "fastest", "fastest train": "fastest",
    "quickest route": "fastest", "minimum time": "fastest", "least time": "fastest",
    "direct train": "fastest",
}

INTENT_PHRASES: Dict[str, List[str]] = {
    "BOOKING_HISTORY": ["show my bookings", "my bookings", "booking history",
        "show my booking", "list bookings", "list my bookings", "all my bookings",
        "view bookings", "show bookings", "view my bookings", "my reservations"],
    "CANCEL_BOOKING": ["cancel booking", "cancel my booking", "cancel ticket",
        "cancel my ticket", "cancel reservation", "cancellation", "cancel",
        "i want to cancel"],
    "BOOK_TICKET": ["book ticket", "book tickets", "reserve ticket", "reserve tickets",
        "buy ticket", "purchase ticket", "book me a ticket", "i want a ticket",
        "i need a ticket", "get me a ticket", "book a train", "book train",
        "i want to book", "make a booking", "reserve a seat"],
    "ROUTE_SEARCH": ["find train", "find trains", "search train", "search trains",
        "trains from", "train from", "is there a train", "any train",
        "trains between", "train between", "show trains", "list trains",
        "available trains", "find me a train", "trains to", "what trains",
        "which trains", "any trains"],
    "FARE_ESTIMATE": ["fare from", "fare between", "fare to", "how much does it cost",
        "how much is the fare", "how much is the ticket", "what is the fare",
        "ticket price", "check fare", "ticket cost", "how much", "price from", "cost from"],
    "COMPARE_ROUTES": ["compare routes", "compare trains", "which is better",
        "better route", "best route", "route comparison"],
    "FASTEST_ROUTE": ["fastest route", "quickest route", "fastest train", "quickest train"],
    "CHEAPEST_ROUTE": ["cheapest route", "cheapest train", "budget train", "low cost train"],
    "SHORTEST_ROUTE": ["shortest route", "fewest stops"],
    "CHECK_ROUTE": ["route for", "route for train", "show route", "check route",
        "train route", "schedule for", "show schedule", "train stops",
        "stations on", "where does train stop"],
}

REQUIRED_SLOTS: Dict[str, List[str]] = {
    "ROUTE_SEARCH": ["source", "destination", "date"],
    "BOOK_TICKET": ["source", "destination", "date", "travel_class", "passengers"],
    "FARE_ESTIMATE": ["source", "destination"],
    "CANCEL_BOOKING": ["booking_id"],
    "CHECK_ROUTE": ["train_number"],
    "SHORTEST_ROUTE": ["source", "destination", "date"],
    "CHEAPEST_ROUTE": ["source", "destination", "date"],
    "FASTEST_ROUTE": ["source", "destination", "date"],
    "COMPARE_ROUTES": ["source", "destination"],
}

CLARIFICATION_QUESTIONS: Dict[str, str] = {
    "source": "Where are you traveling from?",
    "destination": "Where are you traveling to?",
    "date": "When do you want to travel? (e.g. tomorrow, 15 June, 20/06)",
    "travel_class": "Which class? (Sleeper, 3AC, 2AC, 1AC, Chair Car, 2S, GN)",
    "passengers": "How many passengers?",
    "budget": "What's your budget? (e.g. ₹1000)",
    "booking_id": "Please provide your Booking ID.",
    "train_number": "Please provide the train number.",
}

# =============================================================================
# Helpers
# =============================================================================

INDIAN_TIMEZONE = ZoneInfo("Asia/Kolkata")


@dataclass
class StationCandidate:
    code: str
    text: str
    start: int
    end: int
    role: Optional[str] = None


class ChatNLPService:
    """
    Flexible Indian Railways conversational NLP service.

    Improvements over a rigid parser:
    - expands contractions (I'm -> I am) and chat abbreviations (pls -> please)
    - handles typos with fuzzy matching on stations, classes, and intents
    - detects corrections: "no, from Delhi" overwrites previous source
    - richer entity extraction: "me and my friend" -> 2 passengers, "under ₹500" -> budget constraint
    - hybrid intent detection: phrase matchers + keyword scoring + vector similarity
    - robust date/time parser for casual expressions ("in 2 days", "this weekend")
    """

    def __init__(self) -> None:
        self._has_vectors = False
        self.nlp = None

        for model_name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
            try:
                self.nlp = spacy.load(model_name)
                self._has_vectors = self.nlp.vocab.vectors.shape[0] > 0
                break
            except OSError:
                continue

        if self.nlp is None:
            raise RuntimeError(
                "No SpaCy model found. Install one with: python -m spacy download en_core_web_sm"
            )

        self._setup_entity_ruler()
        self._setup_phrase_matchers()
        self._setup_token_matchers()

        self._intent_anchors: Optional[Dict[str, Any]] = None
        if self._has_vectors:
            self._intent_anchors = {
                intent: self.nlp(phrase)
                for intent, phrase in {
                    "ROUTE_SEARCH": "find trains from one city to another",
                    "BOOK_TICKET": "book a train ticket",
                    "CANCEL_BOOKING": "cancel my booking reservation",
                    "BOOKING_HISTORY": "show all my bookings",
                    "FARE_ESTIMATE": "what is the fare price cost",
                    "CHECK_ROUTE": "show route stops for a train",
                    "COMPARE_ROUTES": "compare route options",
                    "FASTEST_ROUTE": "fastest route with minimum time",
                    "CHEAPEST_ROUTE": "cheapest route with minimum fare",
                    "SHORTEST_ROUTE": "shortest route with fewest stops",
                }.items()
            }

        # Prepare fuzzy‑friendly lists for fallback intent matching
        self._intent_keywords: Dict[str, List[str]] = {
            intent: [re.sub(r"\s+", " ", p).lower() for p in phrases]
            for intent, phrases in INTENT_PHRASES.items()
        }
        self._all_station_names = sorted(set(STATION_ALIASES.keys()), key=len, reverse=True)
        self._all_station_codes = sorted(set(STATION_ALIASES.values()))
        self._station_code_lookup = {v: v for v in self._all_station_codes}
        self._class_names = sorted(CLASS_ALIASES.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_entity_ruler(self) -> None:
        ruler = self.nlp.add_pipe(
            "entity_ruler",
            after="ner",
            config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True},
        )
        ruler.add_patterns([
            {"label": "STATION", "pattern": alias, "id": code}
            for alias, code in STATION_ALIASES.items()
        ])

    def _setup_phrase_matchers(self) -> None:
        self._intent_pm = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for intent, phrases in INTENT_PHRASES.items():
            self._intent_pm.add(intent, [self.nlp.make_doc(p) for p in phrases])

        self._class_pm = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for alias in sorted(CLASS_ALIASES, key=len, reverse=True):
            self._class_pm.add(alias, [self.nlp.make_doc(alias)])

        self._pref_pm = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for phrase, pref_label in PREFERENCE_PHRASES.items():
            self._pref_pm.add(pref_label, [self.nlp.make_doc(phrase)])

    def _setup_token_matchers(self) -> None:
        m = Matcher(self.nlp.vocab)

        pax_nouns = [
            "ticket", "tickets", "seat", "seats", "berth", "berths",
            "passenger", "passengers", "person", "people", "pax",
            "adult", "adults", "child", "children", "senior", "seniors",
        ]
        m.add("PASSENGER_COUNT", [[{"LIKE_NUM": True}, {"LOWER": {"IN": pax_nouns}}]])
        m.add("PASSENGER_COUNT_FOR", [[{"LOWER": "for"}, {"LIKE_NUM": True}, {"LOWER": {"IN": pax_nouns}}]])
        m.add("PASSENGER_BOOK_N", [[{"LOWER": {"IN": ["book", "reserve"]}}, {"LIKE_NUM": True}]])
        m.add("BOOKING_ID", [
            [{"LOWER": "booking"}, {"IS_DIGIT": True}],
            [{"LOWER": "booking"}, {"LOWER": "id"}, {"IS_DIGIT": True}],
            [{"LOWER": {"IN": ["cancel", "cancellation"]}}, {"IS_DIGIT": True}],
            [{"ORTH": "#"}, {"IS_DIGIT": True}],
        ])
        m.add("TRAIN_NUMBER", [
            [{"LOWER": "train"}, {"IS_DIGIT": True, "LENGTH": {">=": 4, "<=": 6}}],
            [{"LOWER": "train"}, {"ORTH": "#"}, {"IS_DIGIT": True}],
        ])
        m.add("BUDGET", [
            [{"ORTH": "₹"}, {"LIKE_NUM": True}],
            [{"LOWER": {"IN": ["rs", "rs.", "inr"]}}, {"LIKE_NUM": True}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["rupees", "inr"]}}],
        ])
        # New: "under 500", "less than 500", "more than 500", "around 500"
        m.add("BUDGET_UNDER", [[{"LOWER": {"IN": ["under", "below", "less", "upto", "within"]}}, {"LIKE_NUM": True}]])
        m.add("BUDGET_ABOVE", [[{"LOWER": {"IN": ["above", "more", "over"]}}, {"LIKE_NUM": True}]])
        m.add("BUDGET_AROUND", [[{"LOWER": {"IN": ["around", "approx", "approximately", "nearly", "about"]}}, {"LIKE_NUM": True}]])
        # Group of people: "family of 4", "group of 5", "couple", "me and my friend"
        m.add("PASSENGER_GROUP", [
            [{"LOWER": {"IN": ["family", "group", "team", "gang"]}}, {"LOWER": "of"}, {"LIKE_NUM": True}],
        ])
        m.add("PASSENGER_ME_AND", [
            [{"LOWER": "me"}, {"LOWER": "and"}, {"LOWER": {"IN": ["my", "friend", "wife", "husband", "brother", "sister", "friend"]}}],
        ])
        m.add("PASSENGER_COUPLE", [[{"LOWER": {"IN": ["couple", "pair"]}}]])
        m.add("PASSENGER_WITH", [
            [{"LOWER": "with"}, {"LOWER": {"IN": ["my", "a", "one"]}}, {"LOWER": "friend"}],
            [{"LOWER": "with"}, {"LOWER": "my"}, {"LOWER": {"IN": ["family", "friends"]}}],
        ])

        self._matcher = m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, request: ChatAnalysisRequest) -> ChatAnalysisResponse:
        user_message = self._normalize_text(request.user_message or "")
        # Expand casual English before NLP
        user_message = self._expand_contractions(user_message)
        user_message = self._expand_chat_abbreviations(user_message)
        history = request.conversation_history or []

        memory = self._build_memory(history)
        doc = self.nlp(user_message)
        current_entities = self._extract_entities_from_doc(doc, memory)

        # Detect and apply corrections (e.g., "no, from Delhi")
        corrections = self._detect_corrections(doc, memory)
        resolved = self._merge_context(memory, current_entities.model_dump(exclude_none=True))
        for slot, value in corrections.items():
            resolved[slot] = value
            # If correcting a station, remove the opposite station if ambiguous
            if slot == "source" and "destination" in resolved:
                resolved.pop("destination", None)
            elif slot == "destination" and "source" in resolved:
                resolved.pop("source", None)

        public = {k: v for k, v in resolved.items() if not k.startswith("_")}

        intent = self._detect_intent(doc, public, memory)
        if intent == "UNKNOWN":
            intent = self._infer_intent_from_context(doc, public, memory, current_entities)

        missing = self._check_missing_slots(intent, public)
        pending = memory.get("_pending_slot")
        filled = set(self._filled_slots(current_entities))
        if pending and pending not in filled and pending not in missing:
            missing = [pending] + missing

        clarification_needed = bool(missing)
        clarification_question = CLARIFICATION_QUESTIONS.get(missing[0]) if missing else None
        next_action = self._determine_next_action(intent, clarification_needed)
        action_payload = self._build_action_payload(intent, public, next_action)
        confidence = self._calculate_confidence(intent, current_entities, missing)

        return ChatAnalysisResponse(
            intent=intent,
            confidence=confidence,
            entities=current_entities,
            resolved_context=public,
            missing_required_slots=missing,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
            next_action=next_action,
            action_payload=action_payload,
            memory_patch=current_entities.model_dump(exclude_none=True),
        )

    # ------------------------------------------------------------------
    # Text normalisation & chat expansion
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("—", " ").replace("–", " ").replace("/", " / ")
        text = text.replace("&", " and ")
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        text = re.sub(r"[?!]{2,}", lambda m: m.group(0)[0], text)  # collapse multiple ?! to one
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions so spaCy can parse them."""
        contractions = {
            "i'm": "i am", "im": "i am", "you're": "you are", "youre": "you are",
            "he's": "he is", "she's": "she is", "it's": "it is", "its": "it is",
            "we're": "we are", "they're": "they are", "i've": "i have",
            "you've": "you have", "we've": "we have", "they've": "they have",
            "i'd": "i would", "you'd": "you would", "he'd": "he would",
            "she'd": "she would", "we'd": "we would", "they'd": "they would",
            "i'll": "i will", "you'll": "you will", "he'll": "he will",
            "she'll": "she will", "we'll": "we will", "they'll": "they will",
            "can't": "cannot", "cant": "cannot", "don't": "do not", "dont": "do not",
            "won't": "will not", "wont": "will not", "shouldn't": "should not",
            "couldn't": "could not", "wouldn't": "would not", "isn't": "is not",
            "aren't": "are not", "doesn't": "does not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "hadn't": "had not", "mightn't": "might not", "mustn't": "must not",
            "wanna": "want to", "gonna": "going to", "gotta": "got to",
            "hafta": "have to", "needta": "need to", "coulda": "could have",
            "woulda": "would have", "shoulda": "should have", "mighta": "might have",
        }
        pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions.keys()) + r')\b', re.IGNORECASE)
        return pattern.sub(lambda m: contractions[m.group(0).lower()], text)

    def _expand_chat_abbreviations(self, text: str) -> str:
        """Expand common English chat abbreviations (no Hinglish)."""
        abbreviations = {
            "pls": "please", "plz": "please", "u": "you", "r": "are",
            "ur": "your", "tmrw": "tomorrow", "2moro": "tomorrow",
            "thx": "thanks", "ty": "thank you", "btw": "by the way",
            "wt": "what", "ppl": "people", "msg": "message",
            "idk": "i don't know", "np": "no problem", "y": "why",
            "yup": "yes", "nope": "no", "lol": "", "omg": "",
            "brb": "be right back", "gtg": "got to go", "ttyl": "talk to you later",
            "idc": "i don't care", "tbh": "to be honest", "imo": "in my opinion",
            "afk": "away from keyboard", "bff": "best friend forever",
            "fyi": "for your information", "jk": "just kidding", "rofl": "",
        }
        pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in abbreviations.keys()) + r')\b', re.IGNORECASE)
        return pattern.sub(lambda m: abbreviations[m.group(0).lower()], text)

    def _compact_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", self._normalize_text(text).lower()).strip()

    # ------------------------------------------------------------------
    # Memory / history
    # ------------------------------------------------------------------

    def _build_memory(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        memory: Dict[str, Any] = {}
        for msg in history:
            role = str(msg.get("role", ""))
            content = self._normalize_text(str(msg.get("content", "")))
            content = self._expand_contractions(content)
            content = self._expand_chat_abbreviations(content)
            if not content:
                continue

            doc = self.nlp(content)
            src, dst = self._extract_stations_from_doc(doc, memory)
            if src:
                memory["source"] = src
            if dst:
                memory["destination"] = dst

            date = self._extract_date(content)
            if date:
                memory["date"] = date

            cls = self._extract_class_from_doc(doc)
            if cls:
                memory["travel_class"] = cls

            pax = self._extract_passengers_from_doc(doc)
            if pax is not None:
                memory["passengers"] = pax

            if role == "assistant":
                slot = self._guess_pending_slot(content)
                if slot:
                    memory["_pending_slot"] = slot

        return memory

    def _guess_pending_slot(self, assistant_text: str) -> Optional[str]:
        text = self._compact_text(assistant_text)
        slot_signals: Dict[str, List[str]] = {
            "source": ["traveling from", "coming from", "source", "where are you from", "from where"],
            "destination": ["traveling to", "destination", "going to", "where do you want to go", "to where"],
            "date": ["when do you want", "travel date", "which date", "what date"],
            "travel_class": ["which class", "what class", "preferred class", "class would you"],
            "passengers": ["how many passengers", "how many tickets", "number of passengers", "how many people"],
            "budget": ["your budget", "what's your budget", "budget"],
            "booking_id": ["booking id", "booking number", "id number"],
            "train_number": ["train number", "which train", "train no"],
        }
        for slot, signals in slot_signals.items():
            if any(sig in text for sig in signals):
                return slot
        return None

    def _merge_context(self, base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in patch.items():
            if value is not None:
                merged[key] = value
        return merged

    def _detect_corrections(self, doc: Doc, memory: Dict[str, Any]) -> Dict[str, str]:
        """If user says 'no, from Delhi' or 'actually Mumbai', interpret as a correction."""
        tokens = [t.lower_ for t in doc]
        negations = {"no", "not", "nope", "nah", "actually", "wrong", "change", "instead", "correct"}
        if not any(t in negations for t in tokens):
            return {}

        # Find station entities in the correction message
        station_codes = []
        for ent in doc.ents:
            if ent.label_ == "STATION":
                code = ent.kb_id_ or self._resolve_station_name(ent.text)
                if code:
                    station_codes.append((code, ent.root))
        if not station_codes:
            return {}

        # Determine which slot we are likely correcting
        pending = memory.get("_pending_slot")
        if pending in ("source", "destination"):
            return {pending: station_codes[0][0]}

        # If no pending slot, check if the utterance contains 'from' or 'to'
        for code, root in station_codes:
            role = self._station_role_from_deps(root)  # will be None if no prep
            if role in ("source", "destination"):
                return {role: code}

        # If ambiguous, fall back to the station that wasn't in memory
        if memory.get("source") and not memory.get("destination"):
            return {"destination": station_codes[0][0]}
        if memory.get("destination") and not memory.get("source"):
            return {"source": station_codes[0][0]}

        return {}

    # ------------------------------------------------------------------
    # Intent detection (multi‑strategy)
    # ------------------------------------------------------------------

    def _detect_intent(self, doc: Doc, context: Dict[str, Any], memory: Dict[str, Any]) -> str:
        # Strategy 1: PhraseMatcher
        doc_lower = self.nlp.make_doc(doc.text.lower())
        matches = self._intent_pm(doc_lower)
        if matches:
            best = max(matches, key=lambda m: m[2] - m[1])
            return self.nlp.vocab.strings[best[0]]

        # Strategy 2: Vector similarity (if available)
        if self._has_vectors and self._intent_anchors:
            intent = self._intent_by_similarity(doc)
            if intent != "UNKNOWN":
                return intent

        # Strategy 3: Fuzzy keyword scoring against lemmatised tokens
        lemmas = {t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct}
        best_intent = "UNKNOWN"
        best_score = 0
        for intent, phrases in self._intent_keywords.items():
            score = 0
            for phrase in phrases:
                # Fuzzy match of the whole phrase (useful for typos)
                if get_close_matches(doc.text.lower(), [phrase], n=1, cutoff=0.8):
                    score += 1
                # Also count individual keyword overlaps
                phrase_words = set(re.findall(r"\w+", phrase))
                if phrase_words & lemmas:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_intent = intent
        if best_score >= 1:
            return best_intent

        # Strategy 4: Rule‑based fallback (original token‑set logic)
        tokens_lower = {t.lower_ for t in doc}
        station_ents = [e for e in doc.ents if e.label_ == "STATION"]

        if tokens_lower & {"cancel", "cancellation", "void", "drop"}:
            return "CANCEL_BOOKING"
        if tokens_lower & {"fare", "price", "cost", "amount"} and not tokens_lower & {"book", "reserve"}:
            return "FARE_ESTIMATE"
        if tokens_lower & {"compare", "comparison", "better", "best"}:
            return "COMPARE_ROUTES"
        if tokens_lower & {"fastest", "quickest", "minimum time", "least time"}:
            return "FASTEST_ROUTE"
        if tokens_lower & {"cheapest", "budget", "low cost", "lowest fare", "affordable"}:
            return "CHEAPEST_ROUTE"
        if tokens_lower & {"shortest", "fewest", "least stops"}:
            return "SHORTEST_ROUTE"
        if tokens_lower & {"route", "schedule", "stops"} and any(ch.isdigit() for ch in doc.text):
            return "CHECK_ROUTE"
        if len(station_ents) >= 2:
            if tokens_lower & {"book", "reserve", "ticket", "tickets", "seat", "seats"}:
                return "BOOK_TICKET"
            return "ROUTE_SEARCH"
        if tokens_lower & {"book", "reserve", "ticket", "tickets", "seat", "seats"}:
            return "BOOK_TICKET"
        if tokens_lower & {"route", "schedule", "train"}:
            return "ROUTE_SEARCH"

        return "UNKNOWN"

    def _intent_by_similarity(self, doc: Doc) -> str:
        best_intent = "UNKNOWN"
        best_score = 0.62
        for intent, anchor_doc in self._intent_anchors.items():
            score = doc.similarity(anchor_doc)
            if score > best_score:
                best_score = score
                best_intent = intent
        return best_intent

    def _infer_intent_from_context(self, doc: Doc, context: Dict[str, Any], memory: Dict[str, Any], entities: Entity) -> str:
        has_route = bool(context.get("source") and context.get("destination"))
        if not has_route:
            return "UNKNOWN"
        tokens_lower = {t.lower_ for t in doc}
        book_words = {"book", "reserve", "ticket", "tickets", "confirm", "booking", "seat"}
        booking_signals = [
            context.get("travel_class"),
            context.get("passengers"),
            entities.travel_class,
            entities.passengers,
            memory.get("_pending_slot") in {"travel_class", "passengers"},
            bool(tokens_lower & book_words),
        ]
        if any(booking_signals):
            return "BOOK_TICKET"
        time_words = {"today", "tomorrow", "yesterday", "morning", "evening", "night", "week", "noon"}
        date_signals = [
            context.get("date"),
            entities.date,
            memory.get("_pending_slot") == "date",
            bool(tokens_lower & time_words),
        ]
        if any(date_signals):
            return "ROUTE_SEARCH"
        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _extract_entities_from_doc(self, doc: Doc, memory: Dict[str, Any]) -> Entity:
        ent = Entity(
            source=memory.get("source"),
            destination=memory.get("destination"),
            date=memory.get("date"),
            travel_class=memory.get("travel_class"),
            passengers=memory.get("passengers"),
        )

        src, dst = self._extract_stations_from_doc(doc, memory)
        if src:
            ent.source = src
        if dst:
            ent.destination = dst

        via = self._extract_via_stations_from_doc(doc)
        if via:
            ent.via_stations = via

        date = self._extract_date_combined(doc)
        if date:
            ent.date = date

        time = self._extract_time(doc.text)
        if time:
            ent.time = time

        cls = self._extract_class_from_doc(doc)
        if cls:
            ent.travel_class = cls

        pax = self._extract_passengers_from_doc(doc)
        if pax is not None:
            ent.passengers = pax

        budget = self._extract_budget_from_doc(doc)
        if budget is not None:
            ent.budget = budget

        pref = self._extract_preference_from_doc(doc)
        if pref:
            ent.preference = pref

        bid = self._extract_booking_id_from_doc(doc)
        if bid is not None:
            ent.booking_id = bid

        tn = self._extract_train_number_from_doc(doc)
        if tn:
            ent.train_number = tn

        return ent

    # ------------------------------------------------------------------
    # Station extraction (unchanged core, just used by new correction logic)
    # ------------------------------------------------------------------

    def _extract_stations_from_doc(self, doc: Doc, context: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        source: Optional[str] = None
        destination: Optional[str] = None

        station_ents = [e for e in doc.ents if e.label_ == "STATION"]
        candidate_codes = [self._resolve_station_name(e.text) or e.kb_id_ for e in station_ents]
        candidate_codes = [c for c in candidate_codes if c]

        for ent in station_ents:
            code = ent.kb_id_ or self._resolve_station_name(ent.text)
            if not code:
                continue
            role = self._station_role_from_deps(ent)
            if role == "source" and source is None:
                source = code
            elif role == "destination" and destination is None:
                destination = code

        if source is None or destination is None:
            src_r, dst_r = self._extract_stations_regex(doc.text)
            if source is None and src_r:
                source = src_r
            if destination is None and dst_r:
                destination = dst_r

        if source is None and destination is None and len(candidate_codes) == 1:
            code = candidate_codes[0]
            pending = context.get("_pending_slot")
            if pending == "source":
                source = code
            elif pending == "destination":
                destination = code
            elif context.get("source") and not context.get("destination"):
                destination = code
            elif context.get("destination") and not context.get("source"):
                source = code

        if source is None and destination is None and len(candidate_codes) >= 2:
            source, destination = candidate_codes[0], candidate_codes[1]

        if source is None or destination is None:
            gpe_src, gpe_dst = self._extract_stations_from_gpe(doc, source, destination)
            if source is None:
                source = gpe_src
            if destination is None:
                destination = gpe_dst

        return source, destination

    def _station_role_from_deps(self, ent: Span) -> Optional[str]:
        root = ent.root
        if root.dep_ == "pobj":
            prep = root.head.lower_
            if prep == "from":
                return "source"
            if prep in ("to", "towards", "toward"):
                return "destination"
            if prep in ("via", "through", "passing"):
                return "via"
            if prep == "between":
                return "source"
        if root.dep_ == "conj":
            conj_head = root.head
            if conj_head.dep_ == "pobj" and conj_head.head.lower_ == "between":
                return "destination"
        return None

    def _extract_stations_from_gpe(self, doc: Doc, known_source: Optional[str], known_destination: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        source = None
        destination = None
        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC"):
                continue
            code = self._resolve_station_name(ent.text)
            if not code:
                continue
            root = ent.root
            if root.dep_ == "pobj":
                prep = root.head.lower_
                if prep == "from" and known_source is None and source is None:
                    source = code
                elif prep in ("to", "towards") and known_destination is None and destination is None:
                    destination = code
        return source, destination

    def _extract_via_stations_from_doc(self, doc: Doc) -> Optional[List[str]]:
        via: List[str] = []
        for ent in doc.ents:
            if ent.label_ != "STATION":
                continue
            if self._station_role_from_deps(ent) == "via":
                code = ent.kb_id_ or self._resolve_station_name(ent.text)
                if code:
                    via.append(code)
        return via if via else None

    def _extract_stations_regex(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        msg = self._compact_text(text)
        source = destination = None

        m = re.search(r"between\s+([a-z\s]{2,40}?)\s+and\s+([a-z\s]{2,40}?)(?:\s+(?:via|on|for|at|tomorrow|today)|[,.]|$)", msg)
        if m:
            source = self._resolve_station_name(m.group(1).strip())
            destination = self._resolve_station_name(m.group(2).strip())
            if source and destination:
                return source, destination

        m = re.search(r"from\s+([a-z\s]{2,40}?)\s+to\s+([a-z\s]{2,40}?)(?:\s+(?:via|on|for|at|tomorrow|today|in|by)|[,.]|$)", msg)
        if m:
            source = self._resolve_station_name(m.group(1).strip())
            destination = self._resolve_station_name(m.group(2).strip())
            if source and destination:
                return source, destination

        if " to " in msg:
            left, right = msg.split(" to ", 1)
            left_words = left.split()[-4:]
            right_words = right.split()[:4]
            src_c = self._resolve_station_name(" ".join(left_words))
            dst_c = self._resolve_station_name(" ".join(right_words))
            if src_c and dst_c:
                return src_c, dst_c

        return None, None

    def _resolve_station_name(self, name: str) -> Optional[str]:
        name = self._compact_text(name)
        if not name:
            return None
        if name in STATION_ALIASES:
            return STATION_ALIASES[name]
        upper = re.sub(r"\s+", "", name).upper()
        if upper in self._station_code_lookup:
            return upper
        if re.fullmatch(r"[A-Z]{2,6}", upper):
            return upper
        best_code = None
        best_len = 0
        for alias, code in STATION_ALIASES.items():
            if alias in name or name in alias:
                if len(alias) > best_len:
                    best_len = len(alias)
                    best_code = code
        if best_code:
            return best_code
        best_match = get_close_matches(name, self._all_station_names, n=1, cutoff=0.84)
        if best_match:
            return STATION_ALIASES[best_match[0]]
        return None

    # ------------------------------------------------------------------
    # Class / preference
    # ------------------------------------------------------------------

    def _extract_class_from_doc(self, doc: Doc) -> Optional[str]:
        # 1. Exact phrase matcher
        doc_lower = self.nlp.make_doc(doc.text.lower())
        matches = self._class_pm(doc_lower)
        if matches:
            best = max(matches, key=lambda m: m[2] - m[1])
            phrase = self.nlp.vocab.strings[best[0]]
            return CLASS_ALIASES.get(phrase)

        # 2. Fuzzy sliding window over tokens
        tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_space]
        for window_size in (3, 2, 1):
            for i in range(len(tokens) - window_size + 1):
                candidate = " ".join(tokens[i:i+window_size])
                match = get_close_matches(candidate, self._class_names, n=1, cutoff=0.85)
                if match:
                    return CLASS_ALIASES[match[0]]

        # 3. Simple keyword in text
        text = self._compact_text(doc.text)
        for alias in self._class_names:
            if alias in text:
                return CLASS_ALIASES[alias]
        return None

    def _extract_preference_from_doc(self, doc: Doc) -> Optional[str]:
        doc_lower = self.nlp.make_doc(doc.text.lower())
        matches = self._pref_pm(doc_lower)
        if matches:
            best = max(matches, key=lambda m: m[2] - m[1])
            return self.nlp.vocab.strings[best[0]]
        text = self._compact_text(doc.text)
        if any(k in text for k in ["fast", "quick", "soonest"]):
            return "fastest"
        if any(k in text for k in ["cheap", "low cost", "budget", "affordable", "less fare"]):
            return "cheapest"
        if any(k in text for k in ["short", "fewest stops", "least stops"]):
            return "shortest"
        return None

    # ------------------------------------------------------------------
    # Passengers / booking / train / budget (enriched)
    # ------------------------------------------------------------------

    def _extract_passengers_from_doc(self, doc: Doc) -> Optional[int]:
        text = self._compact_text(doc.text)
        word_nums = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "a": 1, "an": 1,
        }
        pax_nouns = {
            "ticket", "tickets", "seat", "seats", "berth", "berths",
            "passenger", "passengers", "person", "people", "pax",
        }

        # Numeric + noun
        for i, token in enumerate(doc):
            if token.lower_ in word_nums:
                nxt = doc[i + 1] if i + 1 < len(doc) else None
                if nxt and nxt.lower_ in pax_nouns:
                    return word_nums[token.lower_]

        # Custom matcher patterns (including new group/couple patterns)
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            if label in ("PASSENGER_COUNT", "PASSENGER_COUNT_FOR", "PASSENGER_BOOK_N"):
                for token in doc[start:end]:
                    if token.like_num and token.text.isdigit():
                        n = int(token.text)
                        if 1 <= n <= 20:
                            return n
            elif label == "PASSENGER_GROUP":
                # "family of 4"
                for token in doc[start:end]:
                    if token.like_num and token.text.isdigit():
                        n = int(token.text)
                        if 1 <= n <= 20:
                            return n
            elif label == "PASSENGER_ME_AND":
                return 2  # "me and my friend"
            elif label == "PASSENGER_COUPLE":
                return 2
            elif label == "PASSENGER_WITH":
                # "with my friend" -> 2, "with my family" -> ambiguous, assume 2
                return 2

        # Common chat shortcuts
        m = re.search(r"\b(\d+)\s*(?:pax|people|persons?|passengers?|tickets?|seats?)\b", text)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 20:
                return n
        m = re.search(r"\b(?:for|book|reserve)\s+(\d+)\b", text)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 20:
                return n
        return None

    def _extract_booking_id_from_doc(self, doc: Doc) -> Optional[int]:
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "BOOKING_ID":
                for token in doc[start:end]:
                    if token.is_digit:
                        return int(token.text)
        text = self._compact_text(doc.text)
        patterns = [
            r"booking\s*(?:id|#|no\.?|number)?\s*:?[\s#-]*(\d+)",
            r"(?:cancel|cancellation|void|drop)\s+(?:booking\s*)?(\d+)",
            r"(?:my\s*)?booking\s*(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return int(m.group(1))
        return None

    def _extract_train_number_from_doc(self, doc: Doc) -> Optional[str]:
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "TRAIN_NUMBER":
                for token in doc[start:end]:
                    if token.is_digit and 4 <= len(token.text) <= 6:
                        return token.text
        text = self._compact_text(doc.text)
        if any(k in text for k in ["train", "route", "schedule", "nr", "no"]):
            m = re.search(r"\b(\d{4,6})\b", text)
            if m:
                return m.group(1)
        return None

    def _extract_budget_from_doc(self, doc: Doc) -> Optional[float]:
        matches = self._matcher(doc)
        budget_value = None
        budget_modifier = "exact"  # under, above, around

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            if label == "BUDGET":
                for token in doc[start:end]:
                    if token.like_num:
                        try:
                            budget_value = float(token.text.replace(",", ""))
                        except ValueError:
                            pass
            elif label == "BUDGET_UNDER":
                budget_modifier = "under"
                for token in doc[start:end]:
                    if token.like_num:
                        try:
                            budget_value = float(token.text.replace(",", ""))
                        except ValueError:
                            pass
            elif label == "BUDGET_ABOVE":
                budget_modifier = "above"
                for token in doc[start:end]:
                    if token.like_num:
                        try:
                            budget_value = float(token.text.replace(",", ""))
                        except ValueError:
                            pass
            elif label == "BUDGET_AROUND":
                budget_modifier = "around"
                for token in doc[start:end]:
                    if token.like_num:
                        try:
                            budget_value = float(token.text.replace(",", ""))
                        except ValueError:
                            pass

        if budget_value is None:
            text = self._compact_text(doc.text)
            for pat in (r"[₹$]\s*(\d[\d,]*(?:\.\d+)?)", r"\b(\d[\d,]*)\s*(?:rupees?|inr|rs\.?|rs)\b"):
                m = re.search(pat, text)
                if m:
                    budget_value = float(m.group(1).replace(",", ""))
                    break

        # For now, we return the raw value; the modifier can be used later (e.g., in action payload)
        # We'll ignore the modifier and just return the number.
        return budget_value

    # ------------------------------------------------------------------
    # Date / time (extended relative expressions)
    # ------------------------------------------------------------------

    def _extract_date_combined(self, doc: Doc) -> Optional[str]:
        for ent in doc.ents:
            if ent.label_ == "DATE":
                parsed = self._parse_spacy_date_text(ent.text)
                if parsed:
                    return parsed
        return self._extract_date(doc.text)

    def _parse_spacy_date_text(self, date_text: str) -> Optional[str]:
        today = datetime.now(INDIAN_TIMEZONE).date()
        text = self._compact_text(date_text)

        # Basic today/tomorrow etc.
        if text in ("today", "tonight", "today night"):
            return today.isoformat()
        if text in ("tomorrow", "tmr", "tmrw", "tomorrow morning", "tomorrow evening", "tomorrow night"):
            return (today + timedelta(days=1)).isoformat()
        if text in ("day after tomorrow", "day after", "overmorrow"):
            return (today + timedelta(days=2)).isoformat()
        if "next week" in text:
            return (today + timedelta(weeks=1)).isoformat()
        if "this week" in text:
            return today.isoformat()
        if "next month" in text:
            return (today.replace(day=1) + timedelta(days=32)).replace(day=1).isoformat()
        if "this month" in text:
            return today.isoformat()

        # "in 3 days", "after 5 days"
        m = re.search(r"(?:in|after)\s+(\d+)\s+days?", text)
        if m:
            days = int(m.group(1))
            return (today + timedelta(days=days)).isoformat()

        # "this weekend" -> next Saturday
        if "this weekend" in text:
            days_until_sat = (5 - today.weekday()) % 7
            if days_until_sat == 0:  # it's already Saturday, take today
                return today.isoformat()
            return (today + timedelta(days=days_until_sat)).isoformat()

        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, day in enumerate(weekdays):
            if f"next {day}" in text:
                delta = (i - today.weekday()) % 7 or 7
                return (today + timedelta(days=delta)).isoformat()
            if f"this {day}" in text or text == day:
                delta = (i - today.weekday()) % 7
                return (today + timedelta(days=delta)).isoformat()

        return None

    def _extract_date(self, text: str) -> Optional[str]:
        today = datetime.now(INDIAN_TIMEZONE).date()
        msg = self._compact_text(text)

        relative_map = {
            "day after tomorrow": 2,
            "tomorrow": 1,
            "tmrw": 1,
            "tmr": 1,
            "today": 0,
            "tonight": 0,
        }
        for key, delta in relative_map.items():
            if key in msg:
                return (today + timedelta(days=delta)).isoformat()

        if "next week" in msg:
            return (today + timedelta(weeks=1)).isoformat()
        if "this week" in msg:
            return today.isoformat()
        if "next month" in msg:
            return (today.replace(day=1) + timedelta(days=32)).replace(day=1).isoformat()

        m = re.search(r"(?:in|after)\s+(\d+)\s+days?", msg)
        if m:
            days = int(m.group(1))
            return (today + timedelta(days=days)).isoformat()

        if "this weekend" in msg:
            days_until_sat = (5 - today.weekday()) % 7
            if days_until_sat == 0:
                return today.isoformat()
            return (today + timedelta(days=days_until_sat)).isoformat()

        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, day in enumerate(weekdays):
            if f"next {day}" in msg:
                delta = (i - today.weekday()) % 7 or 7
                return (today + timedelta(days=delta)).isoformat()
            if f"this {day}" in msg:
                delta = (i - today.weekday()) % 7
                return (today + timedelta(days=delta)).isoformat()
            if msg == day:
                delta = (i - today.weekday()) % 7
                return (today + timedelta(days=delta)).isoformat()

        month_map = {
            "january": 1, "jan": 1, "february": 2, "feb": 2,
            "march": 3, "mar": 3, "april": 4, "apr": 4,
            "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        month_pat = "|".join(sorted(month_map, key=len, reverse=True))
        ordinal = r"(?:st|nd|rd|th)?"

        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", msg)
        if m:
            return m.group(1)

        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", msg)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
            try:
                return datetime(y, mo, d).date().isoformat()
            except ValueError:
                pass

        m = re.search(rf"\b(\d{{1,2}}){ordinal}\s+({month_pat})(?:\s+(\d{{4}}))?\b", msg)
        if m:
            d = int(m.group(1))
            mo = month_map[m.group(2)]
            y = int(m.group(3)) if m.group(3) else today.year
            try:
                date_obj = datetime(y, mo, d).date()
                if not m.group(3) and date_obj < today:
                    date_obj = datetime(y + 1, mo, d).date()
                return date_obj.isoformat()
            except ValueError:
                pass

        m = re.search(rf"\b({month_pat})\s+(\d{{1,2}}){ordinal}(?:\s+(\d{{4}}))?\b", msg)
        if m:
            mo = month_map[m.group(1)]
            d = int(m.group(2))
            y = int(m.group(3)) if m.group(3) else today.year
            try:
                date_obj = datetime(y, mo, d).date()
                if not m.group(3) and date_obj < today:
                    date_obj = datetime(y + 1, mo, d).date()
                return date_obj.isoformat()
            except ValueError:
                pass

        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", msg)
        if m:
            d, mo = int(m.group(1)), int(m.group(2))
            try:
                date_obj = datetime(today.year, mo, d).date()
                if date_obj < today:
                    date_obj = datetime(today.year + 1, mo, d).date()
                return date_obj.isoformat()
            except ValueError:
                pass

        return None

    def _extract_time(self, text: str) -> Optional[str]:
        msg = self._compact_text(text)

        m = re.search(r"\b(\d{1,2}):(\d{2})\s*(?:hrs?|h)?\b", msg)
        if m:
            h, mn = int(m.group(1)), int(m.group(2))
            if 0 <= h < 24 and 0 <= mn < 60:
                return f"{h:02d}:{mn:02d}"

        m = re.search(r"\b(\d{1,2})\s*:\s*(\d{2})\s*(am|pm)\b", msg)
        if m:
            h, mn, mer = int(m.group(1)), int(m.group(2)), m.group(3)
            if 1 <= h <= 12 and 0 <= mn < 60:
                if mer == "pm" and h != 12:
                    h += 12
                elif mer == "am" and h == 12:
                    h = 0
                return f"{h:02d}:{mn:02d}"

        m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", msg)
        if m:
            h, mer = int(m.group(1)), m.group(2)
            if 1 <= h <= 12:
                if mer == "pm" and h != 12:
                    h += 12
                elif mer == "am" and h == 12:
                    h = 0
                return f"{h:02d}:00"

        if any(w in msg for w in ["morning", "morn", "am"]):
            return "08:00"
        if any(w in msg for w in ["afternoon", "noon", "midday"]):
            return "14:00"
        if any(w in msg for w in ["evening", "eve"]):
            return "18:00"
        if any(w in msg for w in ["night", "tonight", "nite"]):
            return "21:00"

        return None

    # ------------------------------------------------------------------
    # Slot / payload / confidence
    # ------------------------------------------------------------------

    def _check_missing_slots(self, intent: str, context: Dict[str, Any]) -> List[str]:
        required = REQUIRED_SLOTS.get(intent, [])
        return [slot for slot in required if not context.get(slot)]

    def _filled_slots(self, entities: Entity) -> List[str]:
        tracked = ["source", "destination", "date", "travel_class", "passengers", "train_number", "booking_id"]
        data = entities.model_dump(exclude_none=True)
        return [k for k in tracked if data.get(k) is not None]

    _INTENT_TO_ACTION: Dict[str, str] = {
        "ROUTE_SEARCH": "SEARCH_ROUTE",
        "SHORTEST_ROUTE": "ROUTE_ANALYSIS",
        "CHEAPEST_ROUTE": "ROUTE_ANALYSIS",
        "FASTEST_ROUTE": "ROUTE_ANALYSIS",
        "COMPARE_ROUTES": "COMPARE_ROUTES",
        "FARE_ESTIMATE": "ESTIMATE_FARE",
        "BOOK_TICKET": "BOOK",
        "BOOKING_HISTORY": "BOOKING_HISTORY",
        "CANCEL_BOOKING": "CANCEL_BOOKING",
        "CHECK_ROUTE": "CHECK_ROUTE",
    }

    def _determine_next_action(self, intent: str, clarification_needed: bool) -> str:
        if clarification_needed:
            return "ASK_CLARIFICATION"
        return self._INTENT_TO_ACTION.get(intent, "UNKNOWN")

    def _build_action_payload(self, intent: str, context: Dict[str, Any], next_action: str) -> Optional[Dict[str, Any]]:
        if next_action in ("ASK_CLARIFICATION", "UNKNOWN"):
            return None

        if next_action in ("SEARCH_ROUTE", "ROUTE_ANALYSIS", "ESTIMATE_FARE", "COMPARE_ROUTES"):
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
        if next_action == "CANCEL_BOOKING":
            return {"booking_id": context.get("booking_id")}
        if next_action == "CHECK_ROUTE":
            return {"train_number": context.get("train_number")}

        return None

    def _calculate_confidence(self, intent: str, entities: Entity, missing_slots: List[str]) -> float:
        base: Dict[str, float] = {
            "BOOK_TICKET": 0.80,
            "ROUTE_SEARCH": 0.80,
            "FARE_ESTIMATE": 0.80,
            "BOOKING_HISTORY": 0.88,
            "CANCEL_BOOKING": 0.85,
            "CHECK_ROUTE": 0.80,
            "SHORTEST_ROUTE": 0.75,
            "CHEAPEST_ROUTE": 0.75,
            "FASTEST_ROUTE": 0.75,
            "COMPARE_ROUTES": 0.70,
            "UNKNOWN": 0.20,
        }
        score = base.get(intent, 0.45)
        score -= min(len(missing_slots) * 0.12, 0.36)
        entity_count = sum(1 for v in entities.model_dump(exclude_none=True).values() if v is not None)
        score += min(entity_count * 0.05, 0.20)
        return round(max(0.10, min(1.00, score)), 2)


# =============================================================================
# Quick-test helper
# =============================================================================

if __name__ == "__main__":
    service = ChatNLPService()
    samples = [
        "is there any train from Bangalore to Delhi",
        "book 2 sleeper tickets tomorrow 4 pm",
        "12th June 4 pm",
        "cancel my booking 9",
        "fare bangalore delhi",
        "no, from Mumbai",  # correction example
        "me and my friend want to go chennai tmrw",  # passengers inference
        "under 500 rs",  # budget constraint
    ]
    for s in samples:
        out = service.analyze(ChatAnalysisRequest(user_message=s))
        print("\nUSER:", s)
        print(out.model_dump())
