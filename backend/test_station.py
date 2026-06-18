from app.repository.station_repo import StationRepository
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.environ.get('DATABASE_URL', 
    'postgresql://railmitra_jvr9_user:vWHmX1BPDusmUwVUUjvKzkIanPxeD7uu@dpg-d8heu342m8qs73b2v1cg-a.oregon-postgres.render.com/railmitra_jvr9')
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
db = Session()

repo = StationRepository()

# Test popular station names
stations = ["Kolkata", "Varanasi", "Delhi", "Chennai", "Mumbai", "Bangalore", "Hyderabad"]
for name in stations:
    code = repo.fuzzy_find_station(name, db)
    print(f"{name} → {code}")