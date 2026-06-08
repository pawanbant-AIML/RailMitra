#!/usr/bin/env python
"""
Populate database with sample Indian Railway data for testing.
Run: python populate_db.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy.orm import Session
from app.models.db import engine, SessionLocal, Base
from app.models.train_models import Station, Train, Route, Fare

# Create all tables
Base.metadata.create_all(bind=engine)

def populate_database():
    db: Session = SessionLocal()
    
    try:
        # Check if already populated
        if db.query(Train).count() > 0:
            print("✓ Database already populated!")
            return
        
        # ──────────────────── STATIONS ────────────────────
        stations = [
            # Bangalore
            Station(station_code="SBC", station_name="Bangalore City Junction", city="Bangalore"),
            Station(station_code="YPR", station_name="Yaswantpur Junction", city="Bangalore"),
            
            # Mumbai
            Station(station_code="CSTM", station_name="Chhatrapati Shivaji Terminus", city="Mumbai"),
            Station(station_code="BCT", station_name="Bombay Central", city="Mumbai"),
            
            # Delhi
            Station(station_code="NDLS", station_name="New Delhi", city="Delhi"),
            Station(station_code="DLI", station_name="Delhi", city="Delhi"),
            
            # Chennai
            Station(station_code="MAS", station_name="Chennai Central", city="Chennai"),
            Station(station_code="MS", station_name="Chennai Central", city="Chennai"),
            
            # Kolkata
            Station(station_code="HWH", station_name="Howrah Junction", city="Kolkata"),
            
            # Pune
            Station(station_code="PUNE", station_name="Pune Junction", city="Pune"),
            
            # Hyderabad
            Station(station_code="SC", station_name="Secunderabad Junction", city="Hyderabad"),
            
            # Jaipur
            Station(station_code="JP", station_name="Jaipur Junction", city="Jaipur"),
        ]
        db.add_all(stations)
        db.commit()
        print(f"✓ Added {len(stations)} stations")
        
        # ──────────────────── TRAINS ────────────────────
        trains = [
            # Bangalore - Mumbai
            Train(train_number="12657", train_name="Shatabdi Express", 
                  source_station_code="SBC", destination_station_code="CSTM"),
            Train(train_number="12952", train_name="Rajdhani Express", 
                  source_station_code="SBC", destination_station_code="CSTM"),
            
            # Delhi - Chennai
            Train(train_number="12434", train_name="New Delhi Chennai Rajdhani", 
                  source_station_code="NDLS", destination_station_code="MAS"),
            Train(train_number="12621", train_name="Tamil Nadu Express", 
                  source_station_code="NDLS", destination_station_code="MAS"),
            
            # Kolkata - Pune
            Train(train_number="12803", train_name="Howrah Pune SF Express", 
                  source_station_code="HWH", destination_station_code="PUNE"),
            
            # Hyderabad - Jaipur
            Train(train_number="12474", train_name="Hyderabad Jaipur Express", 
                  source_station_code="SC", destination_station_code="JP"),
        ]
        db.add_all(trains)
        db.commit()
        print(f"✓ Added {len(trains)} trains")
        
        # ──────────────────── ROUTES ────────────────────
        routes = [
            # Train 12657 Bangalore to Mumbai
            Route(train_number="12657", sequence=1, station_code="SBC", 
                  departure_time="08:00", distance_km=0),
            Route(train_number="12657", sequence=2, station_code="CSTM", 
                  arrival_time="20:30", distance_km=360),
            
            # Train 12952 Bangalore to Mumbai
            Route(train_number="12952", sequence=1, station_code="SBC", 
                  departure_time="18:00", distance_km=0),
            Route(train_number="12952", sequence=2, station_code="CSTM", 
                  arrival_time="06:30", distance_km=360),
            
            # Train 12434 Delhi to Chennai
            Route(train_number="12434", sequence=1, station_code="NDLS", 
                  departure_time="16:00", distance_km=0),
            Route(train_number="12434", sequence=2, station_code="MAS", 
                  arrival_time="08:30", distance_km=2176),
            
            # Train 12621 Delhi to Chennai
            Route(train_number="12621", sequence=1, station_code="NDLS", 
                  departure_time="22:00", distance_km=0),
            Route(train_number="12621", sequence=2, station_code="MAS", 
                  arrival_time="14:30", distance_km=2176),
            
            # Train 12803 Kolkata to Pune
            Route(train_number="12803", sequence=1, station_code="HWH", 
                  departure_time="19:00", distance_km=0),
            Route(train_number="12803", sequence=2, station_code="PUNE", 
                  arrival_time="22:45", distance_km=1464),
            
            # Train 12474 Hyderabad to Jaipur
            Route(train_number="12474", sequence=1, station_code="SC", 
                  departure_time="10:00", distance_km=0),
            Route(train_number="12474", sequence=2, station_code="JP", 
                  arrival_time="22:00", distance_km=950),
        ]
        db.add_all(routes)
        db.commit()
        print(f"✓ Added {len(routes)} routes")
        
        # ──────────────────── FARES ────────────────────
        fares = [
            # Train 12657
            Fare(train_number="12657", class_type="1A", amount=3500),
            Fare(train_number="12657", class_type="2A", amount=2400),
            Fare(train_number="12657", class_type="3A", amount=1600),
            Fare(train_number="12657", class_type="SL", amount=850),
            
            # Train 12952
            Fare(train_number="12952", class_type="1A", amount=4000),
            Fare(train_number="12952", class_type="2A", amount=2800),
            Fare(train_number="12952", class_type="3A", amount=1900),
            Fare(train_number="12952", class_type="SL", amount=1000),
            
            # Train 12434
            Fare(train_number="12434", class_type="1A", amount=5500),
            Fare(train_number="12434", class_type="2A", amount=3800),
            Fare(train_number="12434", class_type="3A", amount=2500),
            Fare(train_number="12434", class_type="SL", amount=1300),
            
            # Train 12621
            Fare(train_number="12621", class_type="1A", amount=5000),
            Fare(train_number="12621", class_type="2A", amount=3500),
            Fare(train_number="12621", class_type="3A", amount=2300),
            Fare(train_number="12621", class_type="SL", amount=1200),
            
            # Train 12803
            Fare(train_number="12803", class_type="2A", amount=3200),
            Fare(train_number="12803", class_type="3A", amount=2100),
            Fare(train_number="12803", class_type="SL", amount=1100),
            
            # Train 12474
            Fare(train_number="12474", class_type="2A", amount=2800),
            Fare(train_number="12474", class_type="3A", amount=1900),
            Fare(train_number="12474", class_type="SL", amount=950),
        ]
        db.add_all(fares)
        db.commit()
        print(f"✓ Added {len(fares)} fares")
        
        print("\n✅ Database populated successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    populate_database()
