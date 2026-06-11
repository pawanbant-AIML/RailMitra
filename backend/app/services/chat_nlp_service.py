"""
enhanced_chat_nlp_service.py  ── v2  (drop-in replacement for chat_nlp_service.py)
═══════════════════════════════════════════════════════════════════════════════════

SpaCy pipeline features used
──────────────────────────────────────────────────────────────────────────────
Feature                   │ Where                       │ Benefit
──────────────────────────┼─────────────────────────────┼──────────────────────
EntityRuler (STATION)     │ _setup_entity_ruler()        │ ~200 Indian stations
  after="ner",            │ _extract_stations_from_doc() │ as named entities;
  overwrite_ents=True,    │                              │ ent.kb_id_ = code
  phrase_matcher_attr=    │                              │ (e.g. "SBC")
  "LOWER"                 │                              │
──────────────────────────┼─────────────────────────────┼──────────────────────
Dependency Parser         │ _station_role_from_deps()    │ Resolves "from X"
                          │                              │ vs "to Y" via
                          │                              │ root.dep_ + head
──────────────────────────┼─────────────────────────────┼──────────────────────
PhraseMatcher  ×3         │ _setup_phrase_matchers()     │ Multi-word, case-free
  - _intent_pm            │ _detect_intent()             │ phrase detection
  - _class_pm             │ _extract_class_from_doc()    │
  - _pref_pm              │ _extract_preference_from_doc │
──────────────────────────┼─────────────────────────────┼──────────────────────
Matcher (token patterns)  │ _setup_token_matchers()      │ Structural token-
  - PASSENGER_COUNT       │ _extract_passengers_from_doc │ level patterns;
  - BOOKING_ID            │ _extract_booking_id_from_doc │ robust to word order
  - TRAIN_NUMBER          │ _extract_train_number_from.. │
  - BUDGET                │ _extract_budget_from_doc()   │
──────────────────────────┼─────────────────────────────┼──────────────────────
Built-in NER  DATE        │ _extract_date_combined()     │ Catches "this
                          │                              │ weekend", "next
                          │                              │ month", etc.
──────────────────────────┼─────────────────────────────┼──────────────────────
Built-in NER  GPE / LOC   │ _extract_stations_from_gpe() │ Fallback for city
                          │                              │ names not in our
                          │                              │ dictionary
──────────────────────────┼─────────────────────────────┼──────────────────────
Word Vectors  (optional)  │ _intent_by_similarity()      │ Semantic intent
  en_core_web_md / lg     │                              │ matching for
  auto-detected           │                              │ out-of-vocab phrases
══════════════════════════════════════════════════════════════════════════════

Upgrade path:
  pip install en-core-web-md   (60 MB, adds word vectors → better intent)
  pip install en-core-web-lg   (750 MB, largest vectors)
  The service auto-detects which model is installed and enables vector
  similarity only when vectors are present (nlp.vocab.vectors.shape[0] > 0).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════
# Schemas  (unchanged – backward compatible)
# ═══════════════════════════════════════════════════════════════════════════

class Entity(BaseModel):
    source: Optional[str] = None
    destination: Optional[str] = None
    via_stations: Optional[List[str]] = None
    date: Optional[str] = None          # ISO-8601 YYYY-MM-DD
    time: Optional[str] = None          # HH:MM  (24-h)
    travel_class: Optional[str] = None  # SL / 2A / 3A / 1A / CC / 2S / GN / EC
    passengers: Optional[int] = None
    budget: Optional[float] = None
    preference: Optional[str] = None    # fastest / cheapest / shortest
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


# ═══════════════════════════════════════════════════════════════════════════
# Station Dictionary  (merged + expanded from both original service files)
# Key  = city name / alias / station code  (always lowercase)
# Value = canonical station code  (uppercase)
# ═══════════════════════════════════════════════════════════════════════════

STATION_ALIASES: Dict[str, str] = {
    # ── Bengaluru / Karnataka ────────────────────────────────────────────
    "bangalore": "SBC",         "bengaluru": "SBC",         "blr": "SBC",
    "bangalore city": "SBC",    "ksr bengaluru": "SBC",     "sbc": "SBC",
    "yesvantpur": "YPR",        "ypr": "YPR",
    "bangalore cantonment": "BAND", "band": "BAND",
    "mysore": "MYS",            "mysuru": "MYS",            "mys": "MYS",
    "hubli": "UBL",             "hubballi": "UBL",          "ubl": "UBL",
    "mangalore": "MAQ",         "mangaluru": "MAQ",         "maq": "MAQ",

    # ── Mumbai / Maharashtra ─────────────────────────────────────────────
    "mumbai": "CSMT",           "bombay": "CSMT",           "bom": "CSMT",
    "csmt": "CSMT",             "cstm": "CSMT",             "vt": "CSMT",
    "mumbai central": "BCT",    "bct": "BCT",
    "lokmanya tilak terminus": "LTT", "ltt": "LTT",
    "dadar": "DR",              "dr": "DR",
    "bandra terminus": "BDTS",  "bdts": "BDTS",
    "pune": "PUNE",             "puna": "PUNE",
    "nashik": "NK",             "nasik": "NK",              "nk": "NK",
    "nagpur": "NGP",            "ngp": "NGP",
    "aurangabad": "AWB",        "awb": "AWB",
    "kolhapur": "KOP",          "kop": "KOP",
    "solapur": "SUR",           "sur": "SUR",
    "akola": "AK",              "ak": "AK",
    "amravati": "AMI",

    # ── Delhi / Rajasthan / Punjab ───────────────────────────────────────
    "delhi": "NDLS",            "new delhi": "NDLS",        "ndls": "NDLS",
    "old delhi": "DLI",         "dli": "DLI",
    "hazrat nizamuddin": "NZM", "nzm": "NZM",
    "anand vihar": "ANVT",      "anvt": "ANVT",
    "jaipur": "JP",             "jp": "JP",
    "jodhpur": "JU",            "ju": "JU",
    "udaipur": "UDZ",           "udz": "UDZ",
    "kota": "KOTA",
    "bikaner": "BKN",           "bkn": "BKN",
    "ajmer": "AII",             "aii": "AII",
    "agra": "AGC",              "agra cantt": "AGC",        "agc": "AGC",
    "amritsar": "ASR",          "asr": "ASR",
    "chandigarh": "CDG",        "cdg": "CDG",
    "ludhiana": "LDH",          "ldh": "LDH",
    "bathinda": "BTI",          "bti": "BTI",

    # ── Chennai / Tamil Nadu ─────────────────────────────────────────────
    "chennai": "MAS",           "madras": "MAS",            "mas": "MAS",
    "chennai central": "MAS",
    "chennai egmore": "MS",     "ms": "MS",
    "coimbatore": "CBE",        "cbe": "CBE",
    "madurai": "MDU",           "mdu": "MDU",
    "tiruchirapalli": "TPJ",    "trichy": "TPJ",            "tpj": "TPJ",
    "tirunelveli": "TEN",       "ten": "TEN",
    "puducherry": "PDY",        "pondicherry": "PDY",       "pdy": "PDY",
    "salem": "SA",              "sa": "SA",
    "erode": "ED",              "ed": "ED",
    "tirupati": "TPTY",         "tpty": "TPTY",
    "chengalpattu": "CGL",
    "vellore": "KPD",

    # ── Kolkata / West Bengal / Odisha / NE ──────────────────────────────
    "kolkata": "HWH",           "calcutta": "HWH",          "howrah": "HWH",
    "hwh": "HWH",
    "sealdah": "SDAH",          "sdah": "SDAH",
    "asansol": "ASN",           "asn": "ASN",
    "dhanbad": "DHN",           "dhn": "DHN",
    "ranchi": "RNC",            "rnc": "RNC",
    "jamshedpur": "TATA",       "tatanagar": "TATA",
    "bhubaneswar": "BBS",       "bbs": "BBS",
    "puri": "PURI",
    "cuttack": "CTC",           "ctc": "CTC",
    "new jalpaiguri": "NJP",    "njp": "NJP",
    "siliguri": "SGUJ",
    "guwahati": "GHY",          "gauhati": "GHY",           "ghy": "GHY",

    # ── Hyderabad / Andhra Pradesh / Telangana ───────────────────────────
    "hyderabad": "HYB",         "hyd": "HYB",               "hyb": "HYB",
    "secunderabad": "SC",       "sc": "SC",
    "kachiguda": "KCG",
    "vijayawada": "BZA",        "bza": "BZA",
    "visakhapatnam": "VSKP",    "vizag": "VSKP",            "vskp": "VSKP",
    "guntur": "GNT",            "gnt": "GNT",
    "rajahmundry": "RJY",       "rajamahendravaram": "RJY", "rjy": "RJY",
    "kakinada": "CCT",          "cct": "CCT",
    "tirupati": "TPTY",

    # ── Gujarat ──────────────────────────────────────────────────────────
    "ahmedabad": "ADI",         "amdavad": "ADI",           "adi": "ADI",
    "vadodara": "BRC",          "baroda": "BRC",            "brc": "BRC",
    "surat": "ST",              "st": "ST",
    "rajkot": "RJT",            "rjt": "RJT",
    "bhavnagar": "BVC",         "bvc": "BVC",
    "gandhinagar": "GNC",

    # ── Uttar Pradesh / Bihar ─────────────────────────────────────────────
    "lucknow": "LKO",           "lko": "LKO",
    "varanasi": "BSB",          "banaras": "BSB",
    "kashi": "BSB",             "bsb": "BSB",
    "patna": "PNBE",            "pnbe": "PNBE",
    "prayagraj": "PRYJ",        "allahabad": "PRYJ",        "pryj": "PRYJ",
    "gorakhpur": "GKP",         "gkp": "GKP",
    "gaya": "GAYA",
    "muzaffarpur": "MFP",       "mfp": "MFP",
    "kanpur": "CNB",            "cawnpore": "CNB",          "cnb": "CNB",
    "agra": "AGC",
    "mathura": "MTJ",           "mtj": "MTJ",
    "meerut": "MTC",
    "bareilly": "BE",           "be": "BE",
    "moradabad": "MB",

    # ── Madhya Pradesh / Chhattisgarh ─────────────────────────────────────
    "bhopal": "BPL",            "bpl": "BPL",
    "gwalior": "GWL",           "gwl": "GWL",
    "jabalpur": "JBP",          "jbp": "JBP",
    "indore": "INDB",           "indb": "INDB",
    "raipur": "R",
    "bilaspur": "BSP",          "bsp": "BSP",

    # ── Kerala ───────────────────────────────────────────────────────────
    "kochi": "ERS",             "cochin": "ERS",
    "ernakulam": "ERS",         "ers": "ERS",
    "thiruvananthapuram": "TVC","trivandrum": "TVC",        "tvc": "TVC",
    "kannur": "CAN",            "cannanore": "CAN",
    "kozhikode": "CLT",         "calicut": "CLT",           "clt": "CLT",
    "thrissur": "TCR",          "trichur": "TCR",           "tcr": "TCR",
    "palakkad": "PGT",          "palghat": "PGT",           "pgt": "PGT",
    "alleppey": "ALLP",         "alappuzha": "ALLP",        "allp": "ALLP",
    "kollam": "QLN",            "quilon": "QLN",            "qln": "QLN",
    "thrissur": "TCR",
}

# ═══════════════════════════════════════════════════════════════════════════
# Travel Class Aliases
# Sorted longest-first so "ac first class" wins over "first" or "ac"
# ═══════════════════════════════════════════════════════════════════════════

CLASS_ALIASES: Dict[str, str] = {
    "executive chair car": "EC",  "executive class": "EC",
    "ac first class": "1A",       "ac 1st class": "1A",
    "first ac": "1A",             "1st ac": "1A",       "ac first": "1A",
    "second ac": "2A",            "2nd ac": "2A",       "ac 2nd class": "2A",
    "2 tier ac": "2A",            "two tier ac": "2A",  "ac 2 tier": "2A",
    "third ac": "3A",             "3rd ac": "3A",       "ac 3rd class": "3A",
    "3 tier ac": "3A",            "three tier ac": "3A","ac three tier": "3A",
    "ac chair car": "CC",         "chair car": "CC",
    "second sitting": "2S",
    "sleeper class": "SL",        "sleeper": "SL",
    "general": "GN",              "unreserved": "GN",
    "ec": "EC",
    "1ac": "1A",  "1a": "1A",
    "2ac": "2A",  "2a": "2A",
    "3ac": "3A",  "3a": "3A",
    "cc": "CC",
    "2s": "2S",
    "sl": "SL",
    "gn": "GN",
    "ac": "2A",   # bare "AC" defaults to 2A
}

# ═══════════════════════════════════════════════════════════════════════════
# Preference Phrases  → canonical preference label
# ═══════════════════════════════════════════════════════════════════════════

PREFERENCE_PHRASES: Dict[str, str] = {
    "shortest route": "shortest",   "minimum stops": "shortest",
    "fewest stops": "shortest",     "least stops": "shortest",
    "shortest path": "shortest",
    "cheapest route": "cheapest",   "cheapest train": "cheapest",
    "least fare": "cheapest",       "budget friendly": "cheapest",
    "low cost": "cheapest",         "most affordable": "cheapest",
    "fastest route": "fastest",     "fastest train": "fastest",
    "quickest route": "fastest",    "minimum time": "fastest",
    "least time": "fastest",        "direct train": "fastest",
}

# ═══════════════════════════════════════════════════════════════════════════
# Intent Phrase Groups  (PhraseMatcher keys)
# ═══════════════════════════════════════════════════════════════════════════

INTENT_PHRASES: Dict[str, List[str]] = {
    "BOOKING_HISTORY": [
        "show my bookings", "my bookings", "booking history",
        "show my booking", "list bookings", "list my bookings",
        "all my bookings", "view bookings", "show bookings",
        "view my bookings", "my reservations",
    ],
    "CANCEL_BOOKING": [
        "cancel booking", "cancel my booking", "cancel ticket",
        "cancel my ticket", "cancel reservation", "cancellation",
        "cancel", "i want to cancel",
    ],
    "BOOK_TICKET": [
        "book ticket", "book tickets", "reserve ticket", "reserve tickets",
        "buy ticket", "purchase ticket", "book me a ticket",
        "i want a ticket", "i need a ticket", "get me a ticket",
        "book a train", "book train", "i want to book",
        "make a booking", "reserve a seat",
    ],
    "ROUTE_SEARCH": [
        "find train", "find trains", "search train", "search trains",
        "trains from", "train from", "is there a train", "any train",
        "trains between", "train between", "show trains", "list trains",
        "available trains", "find me a train", "trains to",
        "what trains", "which trains", "any trains",
    ],
    "FARE_ESTIMATE": [
        "fare from", "fare between", "fare to",
        "how much does it cost", "how much is the fare",
        "how much is the ticket", "what is the fare",
        "ticket price", "check fare", "ticket cost",
        "how much", "price from", "cost from",
    ],
    "COMPARE_ROUTES": [
        "compare routes", "compare trains", "which is better",
        "better route", "best route", "route comparison",
    ],
    "FASTEST_ROUTE":  ["fastest route", "quickest route", "fastest train", "quickest train"],
    "CHEAPEST_ROUTE": ["cheapest route", "cheapest train", "budget train", "low cost train"],
    "SHORTEST_ROUTE": ["shortest route", "fewest stops"],
    "CHECK_ROUTE": [
        "route for", "route for train", "show route", "check route",
        "train route", "schedule for", "show schedule", "train stops",
        "stations on", "where does train stop",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# Required slots per intent  and  clarification questions per slot
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_SLOTS: Dict[str, List[str]] = {
    "ROUTE_SEARCH":    ["source", "destination", "date"],
    "BOOK_TICKET":     ["source", "destination", "date", "travel_class", "passengers"],
    "FARE_ESTIMATE":   ["source", "destination"],
    "CANCEL_BOOKING":  ["booking_id"],
    "CHECK_ROUTE":     ["train_number"],
    "SHORTEST_ROUTE":  ["source", "destination", "date"],
    "CHEAPEST_ROUTE":  ["source", "destination", "date"],
    "FASTEST_ROUTE":   ["source", "destination", "date"],
    "COMPARE_ROUTES":  ["source", "destination"],
}

CLARIFICATION_QUESTIONS: Dict[str, str] = {
    "source":        "Where are you traveling from?",
    "destination":   "Where are you traveling to?",
    "date":          "When do you want to travel? (e.g. tomorrow, 15 June, 20/06)",
    "travel_class":  "Which class? (Sleeper, 3AC, 2AC, 1AC, Chair Car, 2S, GN)",
    "passengers":    "How many passengers?",
    "budget":        "What's your budget? (e.g. ₹1000)",
    "booking_id":    "Please provide your Booking ID.",
    "train_number":  "Please provide the train number.",
}


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced NLP Service
# ═══════════════════════════════════════════════════════════════════════════

class ChatNLPService:
    """
    Indian Railways conversational NLP service.
    Uses SpaCy's full pipeline: EntityRuler, PhraseMatcher, Matcher,
    dependency parser, and built-in NER.
    """

    # ── Initialization ────────────────────────────────────────────────────

    def __init__(self) -> None:
        # Try the largest available model for best vectors/parses
        self._has_vectors = False
        self.nlp = None

        for model_name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
            try:
                self.nlp = spacy.load(model_name)
                self._has_vectors = self.nlp.vocab.vectors.shape[0] > 0
                print(
                    f"✅  SpaCy '{model_name}' loaded"
                    f"  (vectors={'yes' if self._has_vectors else 'no — upgrade to md/lg for semantic intent'})"
                )
                break
            except OSError:
                continue

        if self.nlp is None:
            raise RuntimeError(
                "No SpaCy model found.\n"
                "Install the smallest model with:  python -m spacy download en_core_web_sm\n"
                "Or the best model with:           python -m spacy download en_core_web_md"
            )

        # Pipeline additions
        self._setup_entity_ruler()
        self._setup_phrase_matchers()
        self._setup_token_matchers()

        # Pre-compile the intent anchor docs used for vector similarity
        self._intent_anchors: Optional[Dict[str, Any]] = None
        if self._has_vectors:
            self._intent_anchors = {
                intent: self.nlp(phrase)
                for intent, phrase in {
                    "ROUTE_SEARCH":    "find trains from one city to another",
                    "BOOK_TICKET":     "book a train ticket",
                    "CANCEL_BOOKING":  "cancel my booking reservation",
                    "BOOKING_HISTORY": "show all my bookings",
                    "FARE_ESTIMATE":   "what is the fare price cost",
                    "CHECK_ROUTE":     "show route stops for a train",
                    "FARE_ESTIMATE":   "how much does the ticket cost fare",
                }.items()
            }

    # ── SpaCy Pipeline: EntityRuler ────────────────────────────────────────

    def _setup_entity_ruler(self) -> None:
        """
        Register all Indian station names as STATION entities.

        Placement: AFTER the built-in NER component with overwrite_ents=True.

        Why after NER?
          • NER runs first → adds GPE/LOC for city names, DATE for dates.
          • EntityRuler runs second → upgrades known cities to STATION with
            their precise station code in ent.kb_id_.
          • Unknown cities remain as GPE/LOC (used as a fallback later).
          • DATE entities are untouched since they have no STATION patterns.

        phrase_matcher_attr="LOWER" makes all pattern matching case-insensitive,
        so "BANGALORE", "Bangalore", and "bangalore" all resolve to "SBC".
        """
        ruler = self.nlp.add_pipe(
            "entity_ruler",
            after="ner",
            config={
                "phrase_matcher_attr": "LOWER",
                "overwrite_ents": True,   # STATION overwrites GPE for known cities
            },
        )
        patterns = [
            {"label": "STATION", "pattern": alias, "id": code}
            for alias, code in STATION_ALIASES.items()
        ]
        ruler.add_patterns(patterns)

    # ── SpaCy Pipeline: PhraseMatcher ─────────────────────────────────────

    def _setup_phrase_matchers(self) -> None:
        """
        Three PhraseMatcher instances (attr="LOWER" → case-insensitive):

        _intent_pm  – matches full intent phrases like "show my bookings".
                      Returns the intent label (e.g. "BOOKING_HISTORY") directly.

        _class_pm   – matches travel class aliases like "third ac", "chair car".
                      Patterns sorted longest-first so "ac first class" wins
                      over "first" or "ac".

        _pref_pm    – matches route preference phrases like "shortest route".
                      Returns the normalised label: fastest / cheapest / shortest.
        """
        # Intent phrases
        self._intent_pm = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for intent, phrases in INTENT_PHRASES.items():
            self._intent_pm.add(intent, [self.nlp.make_doc(p) for p in phrases])

        # Class aliases (longest first → most-specific match wins)
        self._class_pm = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for alias in sorted(CLASS_ALIASES, key=len, reverse=True):
            self._class_pm.add(alias, [self.nlp.make_doc(alias)])

        # Preference phrases
        self._pref_pm = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for phrase, pref_label in PREFERENCE_PHRASES.items():
            # Use the label as the matcher key so the label is returned directly
            self._pref_pm.add(pref_label, [self.nlp.make_doc(phrase)])

    # ── SpaCy Pipeline: Matcher (token patterns) ──────────────────────────

    def _setup_token_matchers(self) -> None:
        """
        SpaCy token-level Matcher for structured numerical patterns.

        Patterns use token attributes (LIKE_NUM, IS_DIGIT, LOWER, LENGTH) so
        they are robust to surrounding words and surface variations.

        PASSENGER_COUNT     "2 tickets", "3 passengers"
        PASSENGER_COUNT_FOR "for 4 people", "for 2 passengers"
        PASSENGER_BOOK_N    "book 3", "reserve 2"
        BOOKING_ID          "booking 5", "id 123", "cancel 7"
        TRAIN_NUMBER        "train 12657", "train #12301"
        BUDGET              "₹500", "rs 1000", "1500 rupees"
        """
        m = Matcher(self.nlp.vocab)

        _pax_nouns = [
            "ticket", "tickets", "seat", "seats", "berth", "berths",
            "passenger", "passengers", "person", "people", "pax",
            "adult", "adults", "child", "children", "senior", "seniors",
        ]

        # "2 tickets"  /  "3 passengers"
        m.add("PASSENGER_COUNT", [
            [{"LIKE_NUM": True}, {"LOWER": {"IN": _pax_nouns}}],
        ])
        # "for 3 passengers"
        m.add("PASSENGER_COUNT_FOR", [
            [{"LOWER": "for"}, {"LIKE_NUM": True}, {"LOWER": {"IN": _pax_nouns}}],
        ])
        # "book 2"  /  "reserve 3"
        m.add("PASSENGER_BOOK_N", [
            [{"LOWER": {"IN": ["book", "reserve"]}}, {"LIKE_NUM": True}],
        ])

        # "booking 5"  /  "booking id 123"  /  "cancel 7"  /  "# 99"
        m.add("BOOKING_ID", [
            [{"LOWER": "booking"}, {"IS_DIGIT": True}],
            [{"LOWER": "booking"}, {"LOWER": "id"}, {"IS_DIGIT": True}],
            [{"LOWER": {"IN": ["cancel", "cancellation"]}}, {"IS_DIGIT": True}],
            [{"ORTH": "#"}, {"IS_DIGIT": True}],
        ])

        # "train 12657"  /  "train #12301"
        m.add("TRAIN_NUMBER", [
            [{"LOWER": "train"}, {"IS_DIGIT": True, "LENGTH": {">=": 4, "<=": 6}}],
            [{"LOWER": "train"}, {"ORTH": "#"}, {"IS_DIGIT": True}],
        ])

        # "₹ 500"  /  "rs 1000"  /  "inr 750"  /  "1500 rupees"
        m.add("BUDGET", [
            [{"ORTH": "₹"}, {"LIKE_NUM": True}],
            [{"LOWER": {"IN": ["rs", "rs.", "inr"]}}, {"LIKE_NUM": True}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["rupees", "inr"]}}],
        ])

        self._matcher = m

    # ═════════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════════

    def analyze(self, request: ChatAnalysisRequest) -> ChatAnalysisResponse:
        """Main entry point. Processes the user message and returns full analysis."""
        user_message = (request.user_message or "").strip()
        history      = request.conversation_history or []

        # 1. Build memory from conversation history
        memory = self._build_memory(history)

        # 2. Run the full SpaCy pipeline on the ORIGINAL (non-lowercased) text.
        #    Keeping original case gives the dependency parser the best chance
        #    of correctly identifying proper nouns.
        doc = self.nlp(user_message)

        # 3. Extract entities from current message (pre-filled from memory)
        current_entities = self._extract_entities_from_doc(doc, memory)

        # 4. Merge memory + current entities into a single resolved context
        resolved = self._merge_context(
            memory,
            current_entities.model_dump(exclude_none=True)
        )
        public = {k: v for k, v in resolved.items() if not k.startswith("_")}

        # 5. Detect intent
        intent = self._detect_intent(doc, public, memory)
        if intent == "UNKNOWN":
            intent = self._infer_intent_from_context(doc, public, memory, current_entities)

        # 6. Slot filling
        missing = self._check_missing_slots(intent, public)
        pending = memory.get("_pending_slot")
        filled  = set(self._filled_slots(current_entities))
        if pending and pending not in filled and pending not in missing:
            missing = [pending] + missing

        clarification_needed   = bool(missing)
        clarification_question = CLARIFICATION_QUESTIONS.get(missing[0]) if missing else None

        # 7. Build response
        next_action    = self._determine_next_action(intent, clarification_needed)
        action_payload = self._build_action_payload(intent, public, next_action)
        confidence     = self._calculate_confidence(intent, current_entities, missing)

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

    # ═════════════════════════════════════════════════════════════════════
    # Conversation Memory
    # ═════════════════════════════════════════════════════════════════════

    def _build_memory(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Scan all previous messages and accumulate known entity values.
        Also infers _pending_slot from the last assistant message so that
        short follow-up replies ("tomorrow", "SL", "Mumbai") are resolved
        in the correct slot context.
        """
        memory: Dict[str, Any] = {}
        for msg in history:
            role    = str(msg.get("role", ""))
            content = str(msg.get("content", ""))
            doc     = self.nlp(content)

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
        """
        Detect what slot the assistant was asking for in its last message
        by scanning the text for characteristic question phrases.
        """
        text = assistant_text.lower()
        slot_signals: Dict[str, List[str]] = {
            "source":       ["traveling from", "coming from", "source", "where are you from"],
            "destination":  ["traveling to", "destination", "going to", "where do you want to go"],
            "date":         ["when do you want", "travel date", "which date", "what date"],
            "travel_class": ["which class", "what class", "preferred class", "class would you"],
            "passengers":   ["how many passengers", "how many tickets", "number of passengers"],
            "budget":       ["your budget", "what's your budget"],
            "booking_id":   ["booking id", "booking number"],
            "train_number": ["train number", "which train"],
        }
        for slot, signals in slot_signals.items():
            if any(s in text for s in signals):
                return slot
        return None

    def _merge_context(
        self,
        base: Dict[str, Any],
        patch: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in patch.items():
            if value is not None:
                merged[key] = value
        return merged

    # ═════════════════════════════════════════════════════════════════════
    # Intent Detection
    # ═════════════════════════════════════════════════════════════════════

    def _detect_intent(
        self,
        doc: Doc,
        context: Dict[str, Any],
        memory: Dict[str, Any],
    ) -> str:
        """
        Three-layer intent detection:
          1. PhraseMatcher  – exact multi-word phrase match (fastest, most precise)
          2. Word vectors   – semantic similarity (only with en_core_web_md/lg)
          3. Rule-based     – token-level keyword fallback
        """
        # ── Layer 1: PhraseMatcher ────────────────────────────────────────
        # Process a lowercased copy for the phrase matcher
        doc_lower = self.nlp.make_doc(doc.text.lower())
        matches = self._intent_pm(doc_lower)
        if matches:
            # Pick the match with the longest span (most specific phrase wins)
            best = max(matches, key=lambda m: m[2] - m[1])
            return self.nlp.vocab.strings[best[0]]

        # ── Layer 2: Word-vector similarity (md / lg models only) ─────────
        if self._has_vectors and self._intent_anchors:
            intent = self._intent_by_similarity(doc)
            if intent != "UNKNOWN":
                return intent

        # ── Layer 3: Rule-based token keywords ───────────────────────────
        tokens_lower = {t.lower_ for t in doc}
        station_ents = [e for e in doc.ents if e.label_ == "STATION"]

        if tokens_lower & {"cancel", "cancellation"}:
            return "CANCEL_BOOKING"
        if tokens_lower & {"fare", "price", "cost"} and not tokens_lower & {"book", "reserve"}:
            return "FARE_ESTIMATE"
        if len(station_ents) >= 2:
            if tokens_lower & {"book", "reserve", "ticket", "tickets"}:
                return "BOOK_TICKET"
            return "ROUTE_SEARCH"
        if tokens_lower & {"book", "reserve"}:
            return "BOOK_TICKET"
        if "route" in tokens_lower or "schedule" in tokens_lower:
            return "CHECK_ROUTE"

        return "UNKNOWN"

    def _intent_by_similarity(self, doc: Doc) -> str:
        """
        Semantic intent classification using cosine similarity between word
        vectors.  Only active when en_core_web_md or en_core_web_lg is used.

        Computes similarity between the user's message doc and pre-computed
        anchor phrase docs for each intent. The intent whose anchor is most
        similar is returned if it exceeds a conservative threshold (0.62).
        """
        best_intent = "UNKNOWN"
        best_score  = 0.62   # conservative threshold to avoid false positives

        for intent, anchor_doc in self._intent_anchors.items():
            score = doc.similarity(anchor_doc)
            if score > best_score:
                best_score  = score
                best_intent = intent

        return best_intent

    def _infer_intent_from_context(
        self,
        doc: Doc,
        context: Dict[str, Any],
        memory: Dict[str, Any],
        entities: Entity,
    ) -> str:
        """
        When intent is still UNKNOWN and we have partial route context from
        memory, infer whether the user is continuing a booking or search flow.
        """
        has_route = bool(context.get("source") and context.get("destination"))
        if not has_route:
            return "UNKNOWN"

        tokens_lower = {t.lower_ for t in doc}
        book_words   = {"book", "reserve", "ticket", "tickets", "confirm", "booking"}

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

        time_words = {"today", "tomorrow", "yesterday", "morning", "evening", "night", "week"}
        date_signals = [
            context.get("date"),
            entities.date,
            memory.get("_pending_slot") == "date",
            bool(tokens_lower & time_words),
        ]
        if any(date_signals):
            return "ROUTE_SEARCH"

        return "UNKNOWN"

    # ═════════════════════════════════════════════════════════════════════
    # Entity Extraction
    # ═════════════════════════════════════════════════════════════════════

    def _extract_entities_from_doc(
        self,
        doc: Doc,
        memory: Dict[str, Any],
    ) -> Entity:
        """
        Build an Entity object from a processed SpaCy Doc.
        Pre-populate from memory so short follow-up replies carry context
        (e.g. the user says "tomorrow" after being asked for a date).
        Each extractor overwrites the default only if it finds a new value.
        """
        ent = Entity(
            source       = memory.get("source"),
            destination  = memory.get("destination"),
            date         = memory.get("date"),
            travel_class = memory.get("travel_class"),
            passengers   = memory.get("passengers"),
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

    # ── Stations ──────────────────────────────────────────────────────────

    def _extract_stations_from_doc(
        self,
        doc: Doc,
        context: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Four-strategy station extraction:

        Strategy 1  EntityRuler entities + dependency tree role resolution.
                    Walk ent.root.dep_ and ent.root.head to decide
                    "from Bangalore" → source  vs  "to Mumbai" → destination.

        Strategy 2  Regex on raw text for "from X to Y" / "between X and Y"
                    (handles cases where the dependency parser mis-labels
                    proper nouns).

        Strategy 3  Single STATION entity resolved via _pending_slot context
                    (short follow-up like just "Mumbai" after being asked
                    for the destination).

        Strategy 4  GPE/LOC entity fallback for city names not in our
                    dictionary (uses SpaCy's built-in NER).
        """
        source: Optional[str]     = None
        destination: Optional[str]= None

        # ── Strategy 1: EntityRuler STATION entities + dep tree ───────────
        station_ents = [e for e in doc.ents if e.label_ == "STATION"]
        for ent in station_ents:
            code = ent.kb_id_ or STATION_ALIASES.get(ent.text.lower())
            if not code:
                continue
            role = self._station_role_from_deps(ent)
            if role == "source" and source is None:
                source = code
            elif role == "destination" and destination is None:
                destination = code
            # "via" is handled separately in _extract_via_stations_from_doc

        # ── Strategy 2: Regex fallback ────────────────────────────────────
        if source is None or destination is None:
            src_r, dst_r = self._extract_stations_regex(doc.text)
            if source is None and src_r:
                source = src_r
            if destination is None and dst_r:
                destination = dst_r

        # ── Strategy 3: Single unassigned entity + context ────────────────
        if source is None and destination is None and len(station_ents) == 1:
            code = (
                station_ents[0].kb_id_
                or STATION_ALIASES.get(station_ents[0].text.lower())
            )
            if code:
                pending = context.get("_pending_slot")
                if pending == "source":
                    source = code
                elif pending == "destination":
                    destination = code
                elif context.get("source") and not context.get("destination"):
                    destination = code
                elif context.get("destination") and not context.get("source"):
                    source = code

        # ── Strategy 4: Two unassigned entities → position order ──────────
        if source is None and destination is None and len(station_ents) >= 2:
            codes = [
                e.kb_id_ or STATION_ALIASES.get(e.text.lower())
                for e in station_ents[:2]
            ]
            codes = [c for c in codes if c]
            if len(codes) == 2:
                source, destination = codes[0], codes[1]

        # ── Strategy 5: GPE/LOC fallback ──────────────────────────────────
        if source is None or destination is None:
            gpe_src, gpe_dst = self._extract_stations_from_gpe(
                doc, source, destination
            )
            if source is None:
                source = gpe_src
            if destination is None:
                destination = gpe_dst

        return source, destination

    def _station_role_from_deps(self, ent: Span) -> Optional[str]:
        """
        Walk the syntactic dependency tree to determine the role of a
        STATION entity span.

        Common dependency patterns produced by SpaCy en_core_web_*:

          "from Bangalore"          root.dep_=pobj,  root.head.lower_="from"
                                    → source

          "to Mumbai"               root.dep_=pobj,  root.head.lower_="to"
                                    → destination

          "between Pune and Hyd"    Pune: root.dep_=pobj, head.lower_="between"
                                    Hyderabad: root.dep_=conj, head=Pune
                                    → source / destination respectively

          "via Nagpur"              root.dep_=pobj,  root.head.lower_="via"
                                    → via

          "Bangalore to Mumbai"     Bangalore: may have dep_=nsubj / compound
                                    Mumbai: root.dep_=pobj, head.lower_="to"
                                    → Bangalore unresolved by deps (caught
                                       by Strategy 2 regex), Mumbai = destination
        """
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
                return "source"   # first of "between X and Y"

        # Second argument of "between X and Y"
        # Hyderabad: dep_=conj, head=Pune (which is dep_=pobj of "between")
        if root.dep_ == "conj":
            conj_head = root.head
            if (
                conj_head.dep_ == "pobj"
                and conj_head.head.lower_ == "between"
            ):
                return "destination"

        return None

    def _extract_stations_from_gpe(
        self,
        doc: Doc,
        known_source: Optional[str],
        known_destination: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        SpaCy's built-in NER often tags city names as GPE (geopolitical entity)
        or LOC even if they are not in our STATION_ALIASES dictionary.
        This method attempts to resolve those GPE/LOC spans to station codes
        via substring matching and uses the dependency tree to assign roles.
        """
        source = None
        destination = None

        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC"):
                continue
            city = ent.text.lower()

            # Direct match first
            code = STATION_ALIASES.get(city)
            if not code:
                # Substring / partial match (longest alias wins)
                best_len = 0
                for alias, c in STATION_ALIASES.items():
                    if (alias in city or city in alias) and len(alias) > best_len:
                        code = c
                        best_len = len(alias)

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
        """Extract via/through stops using the EntityRuler STATION entities."""
        via: List[str] = []
        for ent in doc.ents:
            if ent.label_ != "STATION":
                continue
            role = self._station_role_from_deps(ent)
            if role == "via":
                code = ent.kb_id_ or STATION_ALIASES.get(ent.text.lower())
                if code:
                    via.append(code)
        return via if via else None

    def _extract_stations_regex(
        self,
        text: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Regex fallback when the dependency parser does not cleanly assign
        pobj roles.  Handles the most common surface patterns.
        """
        msg = text.lower()
        source = destination = None

        # "between X and Y"
        m = re.search(
            r"between\s+([a-z\s]{2,25}?)\s+and\s+([a-z\s]{2,25?})"
            r"(?:\s+(?:via|on|for|at|tomorrow|today)|[,.]|$)",
            msg,
        )
        if m:
            source      = self._resolve_station_name(m.group(1).strip())
            destination = self._resolve_station_name(m.group(2).strip())
            if source and destination:
                return source, destination

        # "from X to Y"
        m = re.search(
            r"from\s+([a-z\s]{2,25?}?)\s+to\s+([a-z\s]{2,25?})"
            r"(?:\s+(?:via|on|for|at|tomorrow|today|in|by)|[,.]|$)",
            msg,
        )
        if m:
            source      = self._resolve_station_name(m.group(1).strip())
            destination = self._resolve_station_name(m.group(2).strip())
            if source and destination:
                return source, destination

        # "X to Y"  (bare form, e.g. "Bangalore to Mumbai trains")
        if " to " in msg:
            left, right = msg.split(" to ", 1)
            left_words  = left.split()[-3:]   # last 3 tokens of left side
            right_words = right.split()[:3]   # first 3 tokens of right side
            src_c = self._resolve_station_name(" ".join(left_words))
            dst_c = self._resolve_station_name(right_words[0] if right_words else "")
            if src_c and dst_c:
                return src_c, dst_c

        return None, None

    def _resolve_station_name(self, name: str) -> Optional[str]:
        """
        Map a raw string to a station code.
        Priority: exact alias match → substring match → raw code.
        """
        name = name.strip().lower()
        if not name:
            return None
        if name in STATION_ALIASES:
            return STATION_ALIASES[name]
        # Longest-prefix substring match
        best_code, best_len = None, 0
        for alias, code in STATION_ALIASES.items():
            if alias in name and len(alias) > best_len:
                best_code, best_len = code, len(alias)
        if best_code:
            return best_code
        # Raw station code pattern  (2-6 capital letters)
        upper = re.sub(r"\s+", "", name).upper()
        if re.fullmatch(r"[A-Z]{2,6}", upper):
            return upper
        return None

    # ── Travel Class ──────────────────────────────────────────────────────

    def _extract_class_from_doc(self, doc: Doc) -> Optional[str]:
        """
        Use the PhraseMatcher (_class_pm) to find class aliases.
        Patterns are registered longest-first, so "third ac" wins over bare "ac".
        Returns the canonical class code (SL, 3A, 2A, etc.).
        """
        doc_lower = self.nlp.make_doc(doc.text.lower())
        matches = self._class_pm(doc_lower)
        if not matches:
            return None
        best = max(matches, key=lambda m: m[2] - m[1])
        phrase = self.nlp.vocab.strings[best[0]]   # e.g. "third ac"
        return CLASS_ALIASES.get(phrase)

    # ── Passengers ────────────────────────────────────────────────────────

    def _extract_passengers_from_doc(self, doc: Doc) -> Optional[int]:
        """
        Two-phase passenger count extraction:
          Phase 1  – SpaCy Matcher for structural token patterns.
          Phase 2  – Regex fallback for patterns Matcher may miss.
        Also handles written numbers ("two tickets" → 2).
        """
        WORD_NUMS = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12,
        }
        PAX_NOUNS = {
            "ticket", "tickets", "seat", "seats", "berth", "berths",
            "passenger", "passengers", "person", "people", "pax",
        }

        # Written numbers + passenger noun
        for token in doc:
            if token.lower_ in WORD_NUMS:
                nxt = doc[token.i + 1] if token.i + 1 < len(doc) else None
                if nxt and nxt.lower_ in PAX_NOUNS:
                    return WORD_NUMS[token.lower_]

        # SpaCy Matcher token patterns
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            if label in ("PASSENGER_COUNT", "PASSENGER_COUNT_FOR", "PASSENGER_BOOK_N"):
                for token in doc[start:end]:
                    if token.like_num and token.text.isdigit():
                        n = int(token.text)
                        if 1 <= n <= 20:
                            return n

        # Regex fallback
        text = doc.text.lower()
        regex_patterns = [
            r"\b(\d+)\s+(?:ticket|tickets|seat|seats|berth|berths|passenger|passengers|person|people|pax)\b",
            r"\bfor\s+(\d+)\s+(?:passengers?|persons?|people|pax|tickets?)\b",
            r"\bbook\s+(\d+)\b",
            r"\breserve\s+(\d+)\b",
        ]
        for pat in regex_patterns:
            match = re.search(pat, text)
            if match:
                n = int(match.group(1))
                if 1 <= n <= 20:
                    return n

        return None

    # ── Date ──────────────────────────────────────────────────────────────

    def _extract_date_combined(self, doc: Doc) -> Optional[str]:
        """
        Combine SpaCy DATE entities with regex.

        SpaCy DATE entities are tried first because they catch relative
        expressions that regex alone might miss ("this coming Friday",
        "next week", "this weekend").  The regex handles Indian formats
        (DD/MM/YYYY, "15 June 2026") that SpaCy may not normalise.
        """
        for ent in doc.ents:
            if ent.label_ == "DATE":
                parsed = self._parse_spacy_date_text(ent.text)
                if parsed:
                    return parsed
        return self._extract_date(doc.text)

    def _parse_spacy_date_text(self, date_text: str) -> Optional[str]:
        """Normalise a SpaCy DATE entity string to ISO-8601."""
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        text  = date_text.lower().strip()

        if text in ("today", "tonight", "today night"):
            return today.isoformat()
        if text in ("tomorrow", "tmr", "tmrw", "tomorrow morning", "tomorrow evening"):
            return (today + timedelta(days=1)).isoformat()
        if text in ("day after tomorrow", "day after", "overmorrow"):
            return (today + timedelta(days=2)).isoformat()
        if "next week" in text:
            return (today + timedelta(weeks=1)).isoformat()
        if "this week" in text:
            return today.isoformat()

        weekdays = ["monday", "tuesday", "wednesday", "thursday",
                    "friday", "saturday", "sunday"]
        for i, day in enumerate(weekdays):
            if f"next {day}" in text:
                delta = (i - today.weekday()) % 7 or 7
                return (today + timedelta(days=delta)).isoformat()
            if f"this {day}" in text or text == day:
                delta = (i - today.weekday()) % 7
                return (today + timedelta(days=delta)).isoformat()

        return None  # hand off to regex

    def _extract_date(self, text: str) -> Optional[str]:
        """Regex-based date extraction covering common Indian date formats."""
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        msg   = text.lower()

        if "day after tomorrow" in msg:
            return (today + timedelta(days=2)).isoformat()
        if "tomorrow" in msg:
            return (today + timedelta(days=1)).isoformat()
        if "today" in msg:
            return today.isoformat()
        if "next week" in msg:
            return (today + timedelta(weeks=1)).isoformat()

        weekday_names = ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"]
        for i, day in enumerate(weekday_names):
            if f"next {day}" in msg:
                delta = (i - today.weekday()) % 7 or 7
                return (today + timedelta(days=delta)).isoformat()

        # YYYY-MM-DD
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", msg)
        if m:
            return m.group(1)

        # DD/MM/YYYY or DD-MM-YYYY
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", msg)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
            try:
                return datetime(y, mo, d).date().isoformat()
            except ValueError:
                pass

        # Month map (longest keys first to avoid "may" beating "march")
        month_map = {
            "january": 1,  "jan": 1,  "february": 2, "feb": 2,
            "march": 3,    "mar": 3,  "april": 4,    "apr": 4,
            "may": 5,      "june": 6, "jun": 6,      "july": 7,
            "jul": 7,      "august": 8,"aug": 8,     "september": 9,
            "sept": 9,     "sep": 9,  "october": 10, "oct": 10,
            "november": 11,"nov": 11, "december": 12,"dec": 12,
        }
        month_pat = "|".join(sorted(month_map, key=len, reverse=True))

        # "15 june" / "15 june 2026"
        m = re.search(
            rf"\b(\d{{1,2}})\s+({month_pat})(?:\s+(\d{{4}}))?\b", msg
        )
        if m:
            d  = int(m.group(1))
            mo = month_map[m.group(2)]
            y  = int(m.group(3)) if m.group(3) else today.year
            try:
                date_obj = datetime(y, mo, d).date()
                if not m.group(3) and date_obj < today:
                    date_obj = datetime(y + 1, mo, d).date()
                return date_obj.isoformat()
            except ValueError:
                pass

        # "june 15" / "june 15 2026"
        m = re.search(
            rf"\b({month_pat})\s+(\d{{1,2}})(?:\s+(\d{{4}}))?\b", msg
        )
        if m:
            mo = month_map[m.group(1)]
            d  = int(m.group(2))
            y  = int(m.group(3)) if m.group(3) else today.year
            try:
                date_obj = datetime(y, mo, d).date()
                if not m.group(3) and date_obj < today:
                    date_obj = datetime(y + 1, mo, d).date()
                return date_obj.isoformat()
            except ValueError:
                pass

        # DD/MM  (no year)
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

    # ── Time ──────────────────────────────────────────────────────────────

    def _extract_time(self, text: str) -> Optional[str]:
        """Extract departure/arrival time; returns HH:MM (24-hour)."""
        msg = text.lower()
        m = re.search(r"\b(\d{1,2}):(\d{2})\s*(?:hrs?)?\b", msg)
        if m:
            h, mn = int(m.group(1)), int(m.group(2))
            if 0 <= h < 24 and 0 <= mn < 60:
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
        for word, t in (
            ("morning", "08:00"), ("afternoon", "14:00"),
            ("evening", "18:00"), ("night",    "21:00"),
        ):
            if word in msg:
                return t
        return None

    # ── Budget ────────────────────────────────────────────────────────────

    def _extract_budget_from_doc(self, doc: Doc) -> Optional[float]:
        """Use the token Matcher first, then regex."""
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "BUDGET":
                for token in doc[start:end]:
                    if token.like_num:
                        try:
                            return float(token.text.replace(",", ""))
                        except ValueError:
                            pass
        for pat in (r"[₹$]\s*(\d[\d,]*(?:\.\d+)?)",
                    r"\b(\d[\d,]*)\s*(?:rupees?|inr)\b"):
            m = re.search(pat, doc.text.lower())
            if m:
                return float(m.group(1).replace(",", ""))
        return None

    # ── Preference ────────────────────────────────────────────────────────

    def _extract_preference_from_doc(self, doc: Doc) -> Optional[str]:
        """
        PhraseMatcher (_pref_pm) finds phrases like "shortest route".
        The match label IS the normalised preference: fastest / cheapest / shortest.
        """
        doc_lower = self.nlp.make_doc(doc.text.lower())
        matches = self._pref_pm(doc_lower)
        if not matches:
            return None
        best = max(matches, key=lambda m: m[2] - m[1])
        return self.nlp.vocab.strings[best[0]]

    # ── Booking ID ────────────────────────────────────────────────────────

    def _extract_booking_id_from_doc(self, doc: Doc) -> Optional[int]:
        """Use the Matcher BOOKING_ID pattern, then regex fallback."""
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "BOOKING_ID":
                for token in doc[start:end]:
                    if token.is_digit:
                        return int(token.text)
        for pat in (
            r"booking\s*(?:id|#|no\.?)?\s*:?\s*(\d+)",
            r"cancel(?:\s+booking)?\s+(\d+)",
        ):
            m = re.search(pat, doc.text.lower())
            if m:
                return int(m.group(1))
        return None

    # ── Train Number ──────────────────────────────────────────────────────

    def _extract_train_number_from_doc(self, doc: Doc) -> Optional[str]:
        """Use the Matcher TRAIN_NUMBER pattern, then contextual regex."""
        matches = self._matcher(doc)
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "TRAIN_NUMBER":
                for token in doc[start:end]:
                    if token.is_digit and 4 <= len(token.text) <= 6:
                        return token.text
        # Contextual regex: bare 4-6 digit number when "train / route / schedule" mentioned
        tokens_lower = {t.lower_ for t in doc}
        if tokens_lower & {"train", "route", "schedule"}:
            m = re.search(r"\b(\d{4,6})\b", doc.text)
            if m:
                return m.group(1)
        return None

    # ═════════════════════════════════════════════════════════════════════
    # Slot Filling & Clarification
    # ═════════════════════════════════════════════════════════════════════

    def _check_missing_slots(
        self,
        intent: str,
        context: Dict[str, Any],
    ) -> List[str]:
        required = REQUIRED_SLOTS.get(intent, [])
        return [slot for slot in required if not context.get(slot)]

    def _filled_slots(self, entities: Entity) -> List[str]:
        tracked = [
            "source", "destination", "date", "travel_class",
            "passengers", "train_number", "booking_id",
        ]
        data = entities.model_dump(exclude_none=True)
        return [k for k in tracked if data.get(k) is not None]

    # ═════════════════════════════════════════════════════════════════════
    # Next Action / Payload / Confidence
    # ═════════════════════════════════════════════════════════════════════

    _INTENT_TO_ACTION: Dict[str, str] = {
        "ROUTE_SEARCH":    "SEARCH_ROUTE",
        "SHORTEST_ROUTE":  "ROUTE_ANALYSIS",
        "CHEAPEST_ROUTE":  "ROUTE_ANALYSIS",
        "FASTEST_ROUTE":   "ROUTE_ANALYSIS",
        "COMPARE_ROUTES":  "COMPARE_ROUTES",
        "FARE_ESTIMATE":   "ESTIMATE_FARE",
        "BOOK_TICKET":     "BOOK",
        "BOOKING_HISTORY": "BOOKING_HISTORY",
        "CANCEL_BOOKING":  "CANCEL_BOOKING",
        "CHECK_ROUTE":     "CHECK_ROUTE",
    }

    def _determine_next_action(self, intent: str, clarification_needed: bool) -> str:
        if clarification_needed:
            return "ASK_CLARIFICATION"
        return self._INTENT_TO_ACTION.get(intent, "UNKNOWN")

    def _build_action_payload(
        self,
        intent: str,
        context: Dict[str, Any],
        next_action: str,
    ) -> Optional[Dict[str, Any]]:
        if next_action in ("ASK_CLARIFICATION", "UNKNOWN"):
            return None
        if next_action in ("SEARCH_ROUTE", "ROUTE_ANALYSIS", "ESTIMATE_FARE", "COMPARE_ROUTES"):
            return {
                "source":       context.get("source"),
                "destination":  context.get("destination"),
                "date":         context.get("date"),
                "via_stations": context.get("via_stations"),
                "travel_class": context.get("travel_class"),
                "passengers":   context.get("passengers", 1),
                "preference":   context.get("preference"),
            }
        if next_action == "BOOK":
            return {
                "source":       context.get("source"),
                "destination":  context.get("destination"),
                "date":         context.get("date"),
                "travel_class": context.get("travel_class"),
                "passengers":   context.get("passengers"),
            }
        if next_action == "BOOKING_HISTORY":
            return {}
        if next_action == "CANCEL_BOOKING":
            return {"booking_id": context.get("booking_id")}
        if next_action == "CHECK_ROUTE":
            return {"train_number": context.get("train_number")}
        return None

    def _calculate_confidence(
        self,
        intent: str,
        entities: Entity,
        missing_slots: List[str],
    ) -> float:
        base: Dict[str, float] = {
            "BOOK_TICKET":     0.80, "ROUTE_SEARCH":   0.80,
            "FARE_ESTIMATE":   0.80, "BOOKING_HISTORY":0.88,
            "CANCEL_BOOKING":  0.85, "CHECK_ROUTE":    0.80,
            "SHORTEST_ROUTE":  0.75, "CHEAPEST_ROUTE": 0.75,
            "FASTEST_ROUTE":   0.75, "COMPARE_ROUTES": 0.70,
            "UNKNOWN":         0.20,
        }
        score = base.get(intent, 0.45)
        score -= min(len(missing_slots) * 0.12, 0.36)
        entity_count = sum(
            1 for v in entities.model_dump(exclude_none=True).values()
            if v is not None
        )
        score += min(entity_count * 0.05, 0.20)
        return round(max(0.10, min(1.00, score)), 2)
