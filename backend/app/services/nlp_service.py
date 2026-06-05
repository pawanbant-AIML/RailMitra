"""
Rule-based NLP service — no .pkl files needed.
Handles: intent classification + entity extraction for Indian Railways.
"""
import re
from typing import Tuple, Dict, Optional
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
    "anand vihar": "ANVT",
    # Chennai
    "chennai": "MAS", "madras": "MAS", "mas": "MAS",
    "chennai central": "MAS",
    "chennai egmore": "MS", "ms": "MS",
    # Kolkata / Howrah
    "kolkata": "HWH", "calcutta": "HWH", "howrah": "HWH", "hwh": "HWH",
    "sealdah": "SDAH", "sdah": "SDAH",
    # Hyderabad / Secunderabad
    "hyderabad": "HYB", "sc": "SC", "secunderabad": "SC",
    "kachiguda": "KCG",
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

    # Intent keyword sets
    _CANCEL_KW  = ["cancel booking", "cancel ticket", "cancel my booking", "cancel", "cancellation"]
    _HISTORY_KW = ["my bookings", "booking history", "show my booking", "list booking",
                   "all my booking", "show booking"]
    _FARE_KW    = ["fare", "price", "cost", "how much", "charges", "fee", "rate"]
    _ROUTE_KW   = ["route", "stops", "stopping", "halt", "schedule", "timetable",
                   "time table", "timing"]
    _BOOK_KW    = ["book ticket", "reserve ticket", "buy ticket", "purchase ticket",
                   "book me", "book a ticket", "book 2", "book one", "book two",
                   "book three", "book", "reserve", "i want a ticket", "i need a ticket",
                   "get me a ticket"]
    _SEARCH_KW  = ["find train", "search train", "trains from", "train from",
                   "is there a train", "any train", "trains between", "train between",
                   "show trains", "list trains", "trains available", "available trains",
                   "find me a train"]

    def predict(self, text: str) -> Tuple[str, Dict[str, str]]:
        tl = text.lower().strip()
        entities = self._extract_entities(tl)
        intent   = self._classify_intent(tl)
        return intent, entities

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------
    def _classify_intent(self, tl: str) -> str:
        # Order matters – check more specific patterns first
        for kw in self._CANCEL_KW:
            if kw in tl:
                return "cancel_ticket"
        for kw in self._HISTORY_KW:
            if kw in tl:
                return "booking_history"
        for kw in self._FARE_KW:
            if kw in tl:
                return "check_fare"
        for kw in self._ROUTE_KW:
            if kw in tl:
                return "check_route"
        for kw in self._BOOK_KW:
            if kw in tl:
                return "book_ticket"
        for kw in self._SEARCH_KW:
            if kw in tl:
                return "search_train"
        # Fallback: if we can extract two stations, treat as search
        return "search_train"

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
        if count:
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
        """Try multiple regex patterns to find source and destination."""
        patterns = [
            # "from X to Y" with trailing context
            r'\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+(?:on|in|for|by|via|train|tomorrow|today|'
            r'morning|evening|night|class|sleeper|ac|book|reserve|ticket|please)|[,.]|$)',
            # "between X and Y"
            r'\bbetween\s+(.+?)\s+and\s+(.+?)(?:\s+(?:on|in|for|train|tomorrow|today)|[,.]|$)',
            # "X to Y train / ticket"
            r'\b(.+?)\s+to\s+(.+?)\s+(?:train|ticket|journey|travel)',
            # "X to Y" at sentence level (last resort)
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

        # Partial prefix/suffix match
        for alias, code in STATION_ALIASES.items():
            if name.startswith(alias) or alias.startswith(name):
                return code

        # Contains match
        for alias, code in STATION_ALIASES.items():
            if alias in name:
                return code

        # Looks like a raw station code (2-6 uppercase letters)
        upper = name.upper()
        if re.fullmatch(r'[A-Z]{2,6}', upper):
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

        months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2,
            "march": 3, "mar": 3, "april": 4, "apr": 4,
            "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11,
            "december": 12, "dec": 12,
        }
        month_pat = "|".join(months.keys())

        # "15 june" / "15 june 2026"
        m = re.search(rf'(\d{{1,2}})\s+({month_pat})(?:\s+(\d{{4}}))?', tl)
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
        m = re.search(rf'({month_pat})\s+(\d{{1,2}})(?:\s+(\d{{4}}))?', tl)
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
        m = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', tl)
        if m:
            try:
                y = int(m.group(3))
                if y < 100:
                    y += 2000
                return dt(y, int(m.group(2)), int(m.group(1))).isoformat()
            except ValueError:
                pass

        return None

    def _extract_passenger_count(self, tl: str) -> Optional[int]:
        # Word numbers: "two tickets"
        for word, num in WORD_TO_NUM.items():
            if re.search(r'\b' + word + r'\b', tl):
                return num
        # Digit: "2 tickets", "3 passengers", "book 4"
        m = re.search(r'(\d+)\s*(?:ticket|passenger|seat|berth|person|people|pax)', tl)
        if m:
            return int(m.group(1))
        m = re.search(r'book\s+(\d+)', tl)
        if m:
            return int(m.group(1))
        return None

    def _extract_class(self, tl: str) -> Optional[str]:
        sorted_aliases = sorted(CLASS_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
        for alias, code in sorted_aliases:
            if alias in tl:
                return code
        return None

    def _extract_booking_id(self, tl: str) -> Optional[int]:
        m = re.search(r'(?:booking|id|#|no\.?)\s*:?\s*(\d+)', tl)
        if m:
            return int(m.group(1))
        # bare number after "cancel"
        m = re.search(r'cancel\s+(?:booking\s+)?(\d+)', tl)
        if m:
            return int(m.group(1))
        return None

    def _extract_train_number(self, tl: str) -> Optional[str]:
        # Indian train numbers are 5 digits
        m = re.search(r'\b(\d{5})\b', tl)
        return m.group(1) if m else None