from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from app.models.train_models import Station
from difflib import get_close_matches
from typing import Optional


# City name → list of station codes for that city
# This helps resolve "mumbai" → all mumbai stations, "bangalore" → SBC, etc.
CITY_TO_CODES = {
    "bangalore": ["SBC", "YPR", "BNC", "BNCE"],
    "bengaluru": ["SBC", "YPR", "BNC", "BNCE"],
    "mumbai": ["CSTM", "BCT", "LTT", "DR", "BDTS"],
    "bombay": ["CSTM", "BCT", "LTT", "DR", "BDTS"],
    "delhi": ["NDLS", "DLI", "NZM", "ANVT"],
    "new delhi": ["NDLS"],
    "chennai": ["MAS", "MS"],
    "madras": ["MAS", "MS"],
    "kolkata": ["HWH", "SDAH"],
    "calcutta": ["HWH", "SDAH"],
    "howrah": ["HWH"],
    "hyderabad": ["SC", "HYB", "KCG"],
    "secunderabad": ["SC"],
    "pune": ["PUNE"],
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
    "kochi": ["ERS"],
    "ernakulam": ["ERS"],
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
    "mysore": ["MYS"],
    "mysuru": ["MYS"],
    "tirupati": ["TPTY"],
    "vijayawada": ["BZA"],
}


class StationRepository:
    def search(self, query: str, limit: int, db: Session):
        pattern = f"%{query.lower()}%"
        return (
            db.query(Station)
            .filter(
                or_(
                    Station.station_name.ilike(pattern),
                    Station.station_code.ilike(pattern),
                )
            )
            .limit(limit)
            .all()
        )

    def get_by_code(self, code: str, db: Session):
        return db.query(Station).filter(Station.station_code == code.upper()).first()

    def get_codes_for_city(self, city_name: str) -> list[str]:
        """Return all known station codes for a city name."""
        return CITY_TO_CODES.get(city_name.lower().strip(), [])

    def fuzzy_find_station(self, name: str, db: Session) -> Optional[str]:
        """Map a station name/code (possibly a city name) to its best station_code."""
        name = name.strip()
        if not name:
            return None

        # 1. Exact code match (case-insensitive)
        exact = self.get_by_code(name.upper(), db)
        if exact:
            return exact.station_code

        # 2. City alias lookup – return the primary (first) station code
        city_codes = self.get_codes_for_city(name)
        if city_codes:
            # Verify it exists in DB
            for code in city_codes:
                station = self.get_by_code(code, db)
                if station:
                    return station.station_code
            # Return the first code even if not verified
            return city_codes[0]

        # 3. Exact name match (case-insensitive)
        exact_name = (
            db.query(Station)
            .filter(Station.station_name.ilike(name))
            .first()
        )
        if exact_name:
            return exact_name.station_code

        # 4. Partial name match — "bangalore" matches "Bangalore City Jn"
        partial = (
            db.query(Station)
            .filter(Station.station_name.ilike(f"%{name}%"))
            .first()
        )
        if partial:
            return partial.station_code

        # 5. Fuzzy match on station names
        all_stations = db.query(Station).limit(2000).all()
        names = [s.station_name for s in all_stations]
        matches = get_close_matches(name.strip(), names, n=1, cutoff=0.6)
        if matches:
            best = matches[0]
            station = db.query(Station).filter(Station.station_name == best).first()
            if station:
                return station.station_code

        return None

    def get_all_codes_for_city(self, name: str, db: Session) -> list[str]:
        """Return ALL station codes associated with a city — for broad train search."""
        name = name.strip()

        # City alias
        codes = list(self.get_codes_for_city(name))

        # Also search by partial station name or city column
        partials = (
            db.query(Station)
            .filter(
                or_(
                    Station.station_name.ilike(f"%{name}%"),
                    Station.city.ilike(f"%{name}%") if Station.city is not None else False,
                )
            )
            .limit(20)
            .all()
        )
        for s in partials:
            if s.station_code not in codes:
                codes.append(s.station_code)

        return codes