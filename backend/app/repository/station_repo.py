from __future__ import annotations

from difflib import get_close_matches
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import or_, func
from sqlalchemy.orm import Session

from app.models.train_models import Station

# -----------------------------------------------------------------------------
# City / alias knowledge
# -----------------------------------------------------------------------------
# These are intentionally broad so the system can resolve conversational inputs
# like "Bangalore", "Bengaluru", "Majestic", "Mangalore", etc. to station codes.
CITY_TO_CODES: Dict[str, List[str]] = {
    "bangalore": ["SBC", "YPR", "BNC", "BNCE"],
    "bengaluru": ["SBC", "YPR", "BNC", "BNCE"],
    "blr": ["SBC", "YPR", "BNC", "BNCE"],
    "mumbai": ["CSMT", "CSTM", "BCT", "LTT", "DR", "BDTS"],
    "bombay": ["CSMT", "CSTM", "BCT", "LTT", "DR", "BDTS"],
    "delhi": ["NDLS", "DLI", "NZM", "ANVT"],
    "new delhi": ["NDLS"],
    "chennai": ["MAS", "MS", "PER"],
    "madras": ["MAS", "MS", "PER"],
    "kolkata": ["HWH", "SDAH", "KOAA"],
    "calcutta": ["HWH", "SDAH", "KOAA"],
    "howrah": ["HWH"],
    "hyderabad": ["SC", "HYB", "KCG"],
    "secunderabad": ["SC"],
    "pune": ["PUNE", "PA"],
    "ahmedabad": ["ADI"],
    "jaipur": ["JP"],
    "lucknow": ["LKO", "LJN"],
    "patna": ["PNBE", "PPTA"],
    "bhopal": ["BPL"],
    "nagpur": ["NGP"],
    "surat": ["ST"],
    "vadodara": ["BRC"],
    "baroda": ["BRC"],
    "visakhapatnam": ["VSKP"],
    "vizag": ["VSKP"],
    "coimbatore": ["CBE"],
    "madurai": ["MDU"],
    "kochi": ["ERS", "ERSC"],
    "ernakulam": ["ERS", "ERSC"],
    "thiruvananthapuram": ["TVC"],
    "trivandrum": ["TVC"],
    "guwahati": ["GHY"],
    "bhubaneswar": ["BBS"],
    "varanasi": ["BSB"],
    "amritsar": ["ASR"],
    "chandigarh": ["CDG"],
    "jodhpur": ["JU"],
    "ranchi": ["RNC"],
    "gwalior": ["GWL"],
    "mangalore": ["MAQ", "MAJN"],
    "mangaluru": ["MAQ", "MAJN"],
    "udupi": ["UD"],
    "mysore": ["MYS"],
    "mysuru": ["MYS"],
    "tirupati": ["TPTY"],
    "vijayawada": ["BZA"],
    "hubli": ["UBL"],
    "hubballi": ["UBL"],
    "hassan": ["HAS"],
    "shimoga": ["SMET"],
    "davangere": ["DVG"],
    "goa": ["MAO", "VSG", "SWV"],
}

STATION_ALIASES: Dict[str, str] = {
    # Bangalore region
    "bangalore": "SBC",
    "bengaluru": "SBC",
    "blr": "SBC",
    "majestic": "SBC",
    "krantivira sangolli rayanna": "SBC",
    "sbc": "SBC",
    "yesvantpur": "YPR",
    "yeshwanthpur": "YPR",
    "ypr": "YPR",
    "bengaluru cantonment": "BNC",
    "bnc": "BNC",
    "bangalore cantt": "BNC",
    "bangalore east": "BNCE",
    "bnce": "BNCE",

    # Karnataka / south
    "mysore": "MYS",
    "mysuru": "MYS",
    "mys": "MYS",
    "mangalore": "MAQ",
    "mangaluru": "MAQ",
    "manglore": "MAQ",
    "maq": "MAQ",
    "mangalore junction": "MAJN",
    "majn": "MAJN",
    "hubli": "UBL",
    "hubballi": "UBL",
    "ubl": "UBL",
    "hassan": "HAS",
    "udupi": "UD",
    "goa": "MAO",
    "madgaon": "MAO",
    "madgoan": "MAO",
    "mao": "MAO",
    "vasco da gama": "VSG",
    "vsg": "VSG",
    "shoranur": "SRR",
    "kochi": "ERS",
    "ernakulam": "ERS",
    "ers": "ERS",
    "thiruvananthapuram": "TVC",
    "trivandrum": "TVC",
    "tvc": "TVC",

    # Big cities
    "mumbai": "CSMT",
    "bombay": "CSMT",
    "cstm": "CSMT",
    "csmt": "CSMT",
    "bct": "BCT",
    "ltt": "LTT",
    "delhi": "NDLS",
    "new delhi": "NDLS",
    "ndls": "NDLS",
    "chennai": "MAS",
    "madras": "MAS",
    "mas": "MAS",
    "kolkata": "HWH",
    "howrah": "HWH",
    "hwh": "HWH",
    "hyderabad": "SC",
    "secunderabad": "SC",
    "sc": "SC",
    "pune": "PUNE",
    "ahmedabad": "ADI",
    "jaipur": "JP",
    "lucknow": "LKO",
    "patna": "PNBE",
    "bhopal": "BPL",
    "nagpur": "NGP",
    "surat": "ST",
    "vadodara": "BRC",
    "baroda": "BRC",
    "visakhapatnam": "VSKP",
    "vizag": "VSKP",
    "coimbatore": "CBE",
    "madurai": "MDU",
    "bhubaneswar": "BBS",
    "varanasi": "BSB",
    "amritsar": "ASR",
    "chandigarh": "CDG",
    "jodhpur": "JU",
    "ranchi": "RNC",
    "gwalior": "GWL",
    "tirupati": "TPTY",
    "vijayawada": "BZA",
    "guwahati": "GHY",
}

DEFAULT_RESULT_LIMIT = 20


class StationRepository:
    """Station lookup and fuzzy resolution helpers."""

    def _norm(self, text: str) -> str:
        return (text or "").strip().lower()

    def _safe_station_code(self, station: Station) -> str:
        return (station.station_code or "").upper().strip()

    def search(self, query: str, limit: int, db: Session):
        """Broad station search by name, code, or city."""
        q = self._norm(query)
        if not q:
            return []
        pattern = f"%{q}%"
        return (
            db.query(Station)
            .filter(
                or_(
                    func.lower(Station.station_name).like(pattern),
                    func.lower(Station.station_code).like(pattern),
                    func.lower(Station.city).like(pattern),
                )
            )
            .limit(max(1, limit))
            .all()
        )

    def get_by_code(self, code: str, db: Session):
        """Fetch a station by station code."""
        if not code:
            return None
        return db.query(Station).filter(func.upper(Station.station_code) == code.strip().upper()).first()

    def get_by_name(self, name: str, db: Session):
        """Fetch the first station whose name matches exactly or loosely."""
        q = self._norm(name)
        if not q:
            return None
        exact = (
            db.query(Station)
            .filter(func.lower(Station.station_name) == q)
            .first()
        )
        if exact:
            return exact
        return (
            db.query(Station)
            .filter(func.lower(Station.station_name).like(f"%{q}%"))
            .first()
        )

    def get_codes_for_city(self, city_name: str) -> list[str]:
        """Return known station codes for a city name from the alias table."""
        return list(CITY_TO_CODES.get(self._norm(city_name), []))

    def get_all_codes_for_city(self, name: str, db: Session) -> list[str]:
        """
        Return all station codes associated with a city or station alias.

        This is used by broad train search so a query like "Bangalore to Mangalore"
        can consider all terminals for those cities.
        """
        q = self._norm(name)
        codes: List[str] = []

        # 1) Direct city alias expansion.
        codes.extend(self.get_codes_for_city(q))

        # 2) Direct alias resolution, if the query is itself a station alias.
        alias = STATION_ALIASES.get(q)
        if alias:
            codes.append(alias)

        # 3) Match station city and station name from the DB.
        try:
            candidates = (
                db.query(Station)
                .filter(
                    or_(
                        func.lower(Station.station_name).like(f"%{q}%"),
                        func.lower(Station.city).like(f"%{q}%"),
                        func.upper(Station.station_code).like(f"%{q.upper()}%"),
                    )
                )
                .limit(DEFAULT_RESULT_LIMIT)
                .all()
            )
            for st in candidates:
                code = self._safe_station_code(st)
                if code and code not in codes:
                    codes.append(code)
        except Exception:
            # Keep graceful fallback even when the station table is sparse.
            pass

        # 4) Fallback to the single best station.
        best = self.fuzzy_find_station(name, db)
        if best and best not in codes:
            codes.append(best)

        # Deduplicate while preserving order.
        seen = set()
        deduped: List[str] = []
        for code in codes:
            up = code.upper()
            if up not in seen:
                deduped.append(up)
                seen.add(up)
        return deduped

    def _fuzzy_match_from_rows(self, name: str, rows: Sequence[Station]) -> Optional[str]:
        """Return the best fuzzy match code from a station row list."""
        if not rows:
            return None

        try:
            from rapidfuzz import fuzz  # type: ignore

            def score(st: Station) -> int:
                options = [st.station_name or "", st.station_code or "", st.city or ""]
                return max(fuzz.ratio(self._norm(name), self._norm(opt)) for opt in options if opt)

            best = max(rows, key=score)
            return self._safe_station_code(best) or None
        except Exception:
            names = [s.station_name for s in rows if s.station_name]
            matches = get_close_matches(name, names, n=1, cutoff=0.6)
            if not matches:
                return None
            chosen = matches[0]
            for st in rows:
                if st.station_name == chosen:
                    return self._safe_station_code(st) or None
            return None

    def fuzzy_find_station(self, name: str, db: Session) -> Optional[str]:
        """
        Map a station name, code, or city name to the best station_code.

        Resolution order:
        1. Exact station code
        2. Alias table
        3. Exact station name
        4. Partial station name or city match
        5. Fuzzy name match
        """
        q = self._norm(name)
        if not q:
            return None

        # 1) Exact station code
        exact = self.get_by_code(q, db)
        if exact:
            return self._safe_station_code(exact)

        # 2) Alias / city expansion
        alias = STATION_ALIASES.get(q)
        if alias:
            station = self.get_by_code(alias, db)
            return self._safe_station_code(station) if station else alias

        city_codes = self.get_codes_for_city(q)
        for code in city_codes:
            station = self.get_by_code(code, db)
            if station:
                return self._safe_station_code(station)
        if city_codes:
            return city_codes[0].upper()

        # 3) Exact station name / city
        exact_name = (
            db.query(Station)
            .filter(func.lower(Station.station_name) == q)
            .first()
        )
        if exact_name:
            return self._safe_station_code(exact_name)

        city_match = (
            db.query(Station)
            .filter(func.lower(Station.city) == q)
            .first()
        )
        if city_match:
            return self._safe_station_code(city_match)

        # 4) Partial match
        partials = (
            db.query(Station)
            .filter(
                or_(
                    func.lower(Station.station_name).like(f"%{q}%"),
                    func.lower(Station.city).like(f"%{q}%"),
                )
            )
            .limit(50)
            .all()
        )
        if partials:
            code = self._fuzzy_match_from_rows(q, partials)
            if code:
                return code

        # 5) Fuzzy match across a bounded station set
        try:
            all_stations = db.query(Station).limit(3000).all()
        except Exception:
            all_stations = []

        if all_stations:
            code = self._fuzzy_match_from_rows(q, all_stations)
            if code:
                return code

        return None

    def suggest_stations(self, name: str, db: Session, limit: int = 5) -> List[str]:
        """Return up to N likely matching station codes for clarification prompts."""
        q = self._norm(name)
        if not q:
            return []
        candidates = self.search(q, max(limit * 3, 10), db)
        scores: List[Tuple[int, str]] = []
        try:
            from rapidfuzz import fuzz  # type: ignore

            for st in candidates:
                text = " ".join(filter(None, [st.station_name, st.station_code, st.city]))
                score = fuzz.ratio(q, self._norm(text))
                scores.append((score, self._safe_station_code(st)))
        except Exception:
            for st in candidates:
                text = " ".join(filter(None, [st.station_name, st.station_code, st.city]))
                matches = get_close_matches(q, [self._norm(text)], n=1, cutoff=0.0)
                score = 80 if matches else 0
                scores.append((score, self._safe_station_code(st)))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [code for _, code in scores[:limit] if code]

    def is_known_code(self, code: str, db: Session) -> bool:
        """True if a code exists in the station table."""
        return self.get_by_code(code, db) is not None


__all__ = ["StationRepository", "CITY_TO_CODES", "STATION_ALIASES"]
