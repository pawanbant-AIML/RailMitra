"""
Rule-based NLP service — no .pkl files needed.
Handles: intent classification + entity extraction for Indian Railways.

This version is more flexible and better aligned with the chat flow:
- safer intent detection
- stronger route extraction
- better passenger/class/date parsing
- returns "unknown" instead of forcing a train search
"""

import re
from typing import Tuple, Dict, Optional, List
from datetime import date as dt, timedelta

# ---------------------------------------------------------------------------
# City / common-name → station code lookup
# ---------------------------------------------------------------------------
STATION_ALIASES: Dict[str, str] = {
    # Bangalore / Bengaluru
    "bangalore": "SBC", "bengaluru": "SBC", "blr": "SBC",
    "bangalore city": "SBC", "ksr bengaluru": "SBC",
    "yesvantpur": "YPR", "ypr": "YPR",
    "bangalore cantonment": "BAND", "band": "BAND",

    # Mumbai
    "mumbai": "CSMT", "bombay": "CSMT", "bom": "CSMT",
    "csmt": "CSMT", "vt": "CSMT",
    "mumbai central": "BCT", "bct": "BCT",
    "lokmanya tilak": "LTT", "ltt": "LTT",
    "dadar": "DR", "dr": "DR",
    "bandra": "BVI", "bandra terminus": "BDTS",

    # Delhi
    "delhi": "NDLS", "new delhi": "NDLS", "ndls": "NDLS",
    "old delhi": "DLI", "dli": "DLI",
    "hazrat nizamuddin": "NZM", "nzm": "NZM",
    "anand vihar": "ANVT", "anvt": "ANVT",

    # Chennai
    "chennai": "MAS", "madras": "MAS", "mas": "MAS",
    "chennai central": "MAS",
    "chennai egmore": "MS", "ms": "MS",

    # Kolkata / Howrah
    "kolkata": "HWH", "calcutta": "HWH", "howrah": "HWH", "hwh": "HWH",
    "sealdah": "SDAH", "sdah": "SDAH",

    # Hyderabad / Secunderabad
    "hyderabad": "HYB", "secunderabad": "SC", "sc": "SC",
    "kachiguda": "KCG", "hyb": "HYB", "hyd": "SC",

    # Pune
    "pune": "PUNE",

    # Ahmedabad
    "ahmedabad": "ADI", "amdavad": "ADI", "adi": "ADI",

    # Jaipur
    "jaipur": "JP", "jp": "JP",

    # Lucknow
    "lucknow": "LKO", "lko": "LKO",
    "lucknow nr": "LKO",

    # Patna
    "patna": "PNBE", "pnbe": "PNBE",

    # Bhopal
    "bhopal": "BPL", "bpl": "BPL",

    # Nagpur
    "nagpur": "NGP", "ngp": "NGP",

    # Surat
    "surat": "ST", "st": "ST",

    # Vadodara / Baroda
    "vadodara": "BRC", "baroda": "BRC", "brc": "BRC",

    # Visakhapatnam
    "visakhapatnam": "VSKP", "vizag": "VSKP", "vskp": "VSKP",

    # Coimbatore
    "coimbatore": "CBE", "cbe": "CBE",

    # Madurai
    "madurai": "MDU", "mdu": "MDU",

    # Kochi / Ernakulam
    "kochi": "ERS", "cochin": "ERS", "ernakulam": "ERS", "ers": "ERS",

    # Thiruvananthapuram
    "thiruvananthapuram": "TVC", "trivandrum": "TVC", "tvc": "TVC",

    # Guwahati
    "guwahati": "GHY", "ghy": "GHY",

    # Bhubaneswar
    "bhubaneswar": "BBS", "bbs": "BBS",

    # Raipur
    "raipur": "R",

    # Indore
    "indore": "INDB", "indb": "INDB",

    # Agra
    "agra": "AGC", "agc": "AGC",
    "agra cantt": "AGC",

    # Varanasi
    "varanasi": "BSB", "banaras": "BSB", "kashi": "BSB", "bsb": "BSB",

    # Amritsar
    "amritsar": "ASR", "asr": "ASR",

    # Chandigarh
    "chandigarh": "CDG", "cdg": "CDG",

    # Jodhpur
    "jodhpur": "JU", "ju": "JU",

    # Udaipur
    "udaipur": "UDZ", "udz": "UDZ",

    # Kota
    "kota": "KOTA",

    # Gorakhpur
    "gorakhpur": "GKP", "gkp": "GKP",

    # Prayagraj / Allahabad
    "prayagraj": "PRYJ", "allahabad": "PRYJ", "pryj": "PRYJ",

    # Gwalior
    "gwalior": "GWL", "gwl": "GWL",

    # Jabalpur
    "jabalpur": "JBP", "jbp": "JBP",

    # Mangalore
    "mangalore": "MAQ", "mangaluru": "MAQ", "maq": "MAQ",

    # Mysore
    "mysore": "MYS", "mysuru": "MYS", "mys": "MYS",

    # Hubli
    "hubli": "UBL", "ubl": "UBL",

    # Tirupati
    "tirupati": "TPTY", "tpty": "TPTY",

    # Vijayawada
    "vijayawada": "BZA", "bza": "BZA",

    # Guntur
    "guntur": "GNT", "gnt": "GNT",

    # Aurangabad
    "aurangabad": "AWB", "awb": "AWB",

    # Nashik
    "nashik": "NK", "nasik": "NK", "nk": "NK",

    # Kolhapur
    "kolhapur": "KOP", "kop": "KOP",

    # Solapur
    "solapur": "SUR", "sur": "SUR",

    # Ranchi
    "ranchi": "RNC", "rnc": "RNC",

    # Jamshedpur
    "jamshedpur": "TATA", "tatanagar": "TATA",

    # Dhanbad
    "dhanbad": "DHN", "dhn": "DHN",

    # Muzaffarpur
    "muzaffarpur": "MFP", "mfp": "MFP",

    # Gaya
    "gaya": "GAYA",

    # New Jalpaiguri / Siliguri
    "new jalpaiguri": "NJP", "njp": "NJP", "siliguri": "SGUJ",

    # Asansol
    "asansol": "ASN", "asn": "ASN",

    # Kakinada
    "kakinada": "CCT",

    # Rajahmundry
    "rajahmundry": "RJY", "rajamahendravaram": "RJY",

    # Salem
    "salem": "SA", "sa": "SA",

    # Erode
    "erode": "ED", "ed": "ED",

    # Tiruchirapalli / Trichy
    "tiruchirapalli": "TPJ", "trichy": "TPJ", "tpj": "TPJ",

    # Tirunelveli
    "tirunelveli": "TEN", "ten": "TEN",

    # Puducherry
    "puducherry": "PDY", "pondicherry": "PDY", "pdy": "PDY",

    # Kannur
    "kannur": "CAN", "cannanore": "CAN",

    # Kozhikode
    "kozhikode": "CLT", "calicut": "CLT", "clt": "CLT",

    # Thrissur
    "thrissur": "TCR", "tcr": "TCR",

    # Palakkad
    "palakkad": "PGT", "palghat": "PGT", "pgt": "PGT",

    # Alleppey
    "alleppey": "ALLP", "alappuzha": "ALLP",

    # Kollam
    "kollam": "QLN", "quilon": "QLN", "qln": "QLN",

    # Bilaspur
    "bilaspur": "BSP", "bsp": "BSP",

    # Durg
    "durg": "DURG",
}

WORD_TO_NUM: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

# Sorted longest→shortest so multi-word phrases match before single words
CLASS_ALIASES: Dict[str, str] = {
    "ac first class": "1A",
    "ac 1st class": "1A",
    "first ac": "1A",
    "1st ac": "1A",
    "ac first": "1A",
    "1ac": "1A",
    "1a": "1A",

    "second ac": "2A",
    "2nd ac": "2A",
    "ac 2nd class": "2A",
    "2 tier ac": "2A",
    "two tier ac": "2A",
    "2ac": "2A",
    "2a": "2A",

    "third ac": "3A",
    "3rd ac": "3A",
    "ac 3rd class": "3A",
    "3 tier ac": "3A",
    "three tier ac": "3A",
    "ac three tier": "3A",
    "3ac": "3A",
    "3a": "3A",

    "ac chair car": "CC",
    "chair car": "CC",
    "cc": "CC",
    "executive chair car": "EC",
    "executive class": "EC",
    "ec": "EC",

    "second sitting": "2S",
    "2s": "2S",

    "sleeper class": "SL",
    "sleeper": "SL",
    "sl": "SL",

    "general": "GN",
    "unreserved": "GN",
    "gn": "GN",
}


class NLPService:
    """Pure rule-based NLP – no ML model files required."""

    _CANCEL_KW = [
        "cancel booking", "cancel ticket", "cancel my booking",
        "cancel reservation", "cancel", "cancellation"
    ]
    _HISTORY_KW = [
        "my bookings", "booking history", "show my booking",
        "show my bookings", "list booking", "list bookings",
        "all my booking", "all my bookings", "show booking", "show bookings",
        "view bookings"
    ]
    _FARE_KW = ["fare", "price", "cost", "how much", "charges", "fee", "rate", "ticket price"]
    _ROUTE_KW = ["route", "stops", "halt", "schedule", "timetable", "time table", "timing"]
    _BOOK_KW = [
        "book ticket", "reserve ticket", "buy ticket", "purchase ticket",
        "book me", "book a ticket", "i want a ticket", "i need a ticket",
        "get me a ticket", "reserve", "book"
    ]
    _SEARCH_KW = [
        "find train", "find trains", "search train", "search trains",
        "trains from", "train from", "is there a train", "any train",
        "trains between", "train between", "show trains", "list trains",
        "trains available", "available trains", "find me a train"
    ]
    _COMPARE_KW = ["compare", "which is better", "better route"]
    _FASTEST_KW = ["fastest", "quickest", "minimum time"]
    _CHEAPEST_KW = ["cheapest", "budget", "affordable", "low cost"]
    _SHORTEST_KW = ["shortest", "minimum stops", "fewest stops"]

    def predict(self, text: str) -> Tuple[str, Dict[str, str]]:
        tl = self._normalize(text)
        entities = self._extract_entities(tl)
        intent = self._classify_intent(tl, entities)
        return intent, entities

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------
    def _classify_intent(self, tl: str, entities: Dict[str, str]) -> str:
        # Order matters — more specific first.
        if any(kw in tl for kw in self._CANCEL_KW):
            return "cancel_ticket"

        if any(kw in tl for kw in self._HISTORY_KW):
            return "booking_history"

        # Route / booking / fare are the most common railway actions.
        if any(kw in tl for kw in self._FARE_KW):
            return "check_fare"

        if any(kw in tl for kw in self._ROUTE_KW) or self._looks_like_route_query(tl, entities):
            return "check_route"

        if any(kw in tl for kw in self._BOOK_KW):
            return "book_ticket"

        if any(kw in tl for kw in self._COMPARE_KW):
            return "compare_routes"

        if any(kw in tl for kw in self._FASTEST_KW):
            return "fastest_route"

        if any(kw in tl for kw in self._CHEAPEST_KW):
            return "cheapest_route"

        if any(kw in tl for kw in self._SHORTEST_KW):
            return "shortest_route"

        if any(kw in tl for kw in self._SEARCH_KW):
            return "search_train"

        # If we can identify two stations, it's likely a search request.
        if entities.get("source_station") or entities.get("destination_station"):
            return "search_train"

        return "unknown"

    def _looks_like_route_query(self, tl: str, entities: Dict[str, str]) -> bool:
        if "route for train" in tl:
            return True
        if "show route" in tl or "check route" in tl or "train route" in tl:
            return True
        if "from" in tl and "to" in tl:
            return True
        if "between" in tl and ("and" in tl or entities.get("source_station")):
            return True
        return False

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------
    def _extract_entities(self, tl: str) -> Dict[str, str]:
        ents: Dict[str, str] = {}

        src, dst = self._extract_stations(tl)
        if src:
            ents["source_station"] = src
        if dst:
            ents["destination_station"] = dst

        date_val = self._extract_date(tl)
        if date_val:
            ents["date"] = date_val

        count = self._extract_passenger_count(tl)
        if count is not None:
            ents["passenger_count"] = str(count)

        cls = self._extract_class(tl)
        if cls:
            ents["class_type"] = cls

        bid = self._extract_booking_id(tl)
        if bid is not None:
            ents["booking_id"] = str(bid)

        tn = self._extract_train_number(tl)
        if tn:
            ents["train_number"] = tn

        return ents

    def _extract_stations(self, tl: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Try multiple regex patterns to find source and destination.
        Supports:
        - "from X to Y"
        - "X to Y"
        - "between X and Y"
        - "route for train 12657" (no station extraction)
        """
        patterns = [
            r'\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+(?:on|in|for|by|via|train|tomorrow|today|'
            r'morning|evening|night|class|sleeper|ac|book|reserve|ticket|please)|[,.]|$)',
            r'\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+(?:on|in|for|train|tomorrow|today)|[,.]|$)',
            r'\b(.+?)\s+to\s+(.+?)\s+(?:train|ticket|journey|travel)',
            r'\b(.+?)\s+to\s+(.+?)(?:\s+on\s+|\s+for\s+|[,.]|$)',
        ]

        for pat in patterns:
            m = re.search(pat, tl, re.IGNORECASE)
            if m:
                src_raw = m.group(1).strip().rstrip(" ,.")
                dst_raw = m.group(2).strip().rstrip(" ,.")
                src_code = self._resolve(src_raw)
                dst_code = self._resolve(dst_raw)
                if src_code and dst_code:
                    return src_code, dst_code

        return None, None

    def _resolve(self, name: str) -> Optional[str]:
        """Map a city/station name to a station code."""
        name = name.strip().lower()
        if not name:
            return None

        # Direct exact match
        if name in STATION_ALIASES:
            return STATION_ALIASES[name]

        # Try longest aliases first
        sorted_aliases = sorted(STATION_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)

        # Partial prefix/suffix match
        for alias, code in sorted_aliases:
            if name.startswith(alias) or alias.startswith(name):
                return code

        # Contains match
        for alias, code in sorted_aliases:
            if alias in name:
                return code

        # Looks like a raw station code (2-6 alphanumeric uppercase-ish)
        upper = re.sub(r"\s+", "", name).upper()
        if re.fullmatch(r"[A-Z0-9]{2,6}", upper):
            return upper

        return None

    def _extract_date(self, tl: str) -> Optional[str]:
        today = dt.today()

        if "day after tomorrow" in tl:
            return (today + timedelta(days=2)).isoformat()
        if "tomorrow" in tl:
            return (today + timedelta(days=1)).isoformat()
        if "today" in tl:
            return today.isoformat()
        if "next week" in tl:
            return (today + timedelta(weeks=1)).isoformat()

        weekdays = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
        }
        for day_name, idx in weekdays.items():
            if f"next {day_name}" in tl:
                delta = (idx - today.weekday()) % 7
                if delta == 0:
                    delta = 7
                return (today + timedelta(days=delta)).isoformat()

        months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2,
            "march": 3, "mar": 3, "april": 4, "apr": 4,
            "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        month_pat = "|".join(sorted(months.keys(), key=len, reverse=True))

        # "15 june" / "15 june 2026"
        m = re.search(rf"\b(\d{{1,2}})\s+({month_pat})(?:\s+(\d{{4}}))?\b", tl)
        if m:
            try:
                return dt(
                    int(m.group(3) or today.year),
                    months[m.group(2)],
                    int(m.group(1)),
                ).isoformat()
            except ValueError:
                pass

        # "june 15" / "june 15 2026"
        m = re.search(rf"\b({month_pat})\s+(\d{{1,2}})(?:\s+(\d{{4}}))?\b", tl)
        if m:
            try:
                return dt(
                    int(m.group(3) or today.year),
                    months[m.group(1)],
                    int(m.group(2)),
                ).isoformat()
            except ValueError:
                pass

        # DD/MM/YYYY or DD-MM-YYYY
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", tl)
        if m:
            try:
                y = int(m.group(3))
                if y < 100:
                    y += 2000
                return dt(y, int(m.group(2)), int(m.group(1))).isoformat()
            except ValueError:
                pass

        # DD/MM or DD-MM
        m = re.search(r"\b(\d{1,2})[/-](\d{1,2})\b", tl)
        if m:
            try:
                day = int(m.group(1))
                month = int(m.group(2))
                year = today.year
                d = dt(year, month, day)
                if d < today:
                    d = dt(year + 1, month, day)
                return d.isoformat()
            except ValueError:
                pass

        return None

    def _extract_passenger_count(self, tl: str) -> Optional[int]:
        # Word numbers: "two tickets"
        for word, num in WORD_TO_NUM.items():
            if re.search(r"\b" + re.escape(word) + r"\b", tl):
                return num

        # Digit-based patterns
        patterns = [
            r"\b(\d+)\s*(?:ticket|tickets|passenger|passengers|seat|seats|berth|berths|person|people|pax)\b",
            r"\bbook\s+(\d+)\b",
            r"\bfor\s+(\d+)\s*(?:passenger|passengers|person|people|pax|tickets?)\b",
            r"\b(\d+)\s*(?:adult|adults|child|children|kid|kids|senior|seniors)\b",
            r"\b(\d+)\s*(?:sleeper|sl|1a|2a|3a|cc|2s|ec)\b",
        ]

        for pat in patterns:
            m = re.search(pat, tl)
            if m:
                try:
                    n = int(m.group(1))
                    if 1 <= n <= 20:
                        return n
                except ValueError:
                    pass

        return None

    def _extract_class(self, tl: str) -> Optional[str]:
        sorted_aliases = sorted(CLASS_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
        for alias, code in sorted_aliases:
            if alias in tl:
                return code
        return None

    def _extract_booking_id(self, tl: str) -> Optional[int]:
        patterns = [
            r"(?:booking|id|#|no\.?)\s*:?\s*(\d+)",
            r"cancel\s+(?:booking\s+)?(\d+)",
            r"booking\s+(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, tl)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    pass
        return None

    def _extract_train_number(self, tl: str) -> Optional[str]:
        # Indian train numbers are usually 5 digits, but we keep it a bit flexible.
        m = re.search(r"\b(\d{4,6})\b", tl)
        return m.group(1) if m else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        text = (text or "").lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text
