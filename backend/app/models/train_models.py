from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base

class Train(Base):
    __tablename__ = "trains"

    train_number = Column(String, primary_key=True, index=True, nullable=False)
    train_name = Column(String, nullable=False)
    source_station_code = Column(String, nullable=False)
    destination_station_code = Column(String, nullable=False)

class Station(Base):
    __tablename__ = "stations"

    station_code = Column(String, primary_key=True, index=True, nullable=False)
    station_name = Column(String, nullable=False)
    city = Column(String, nullable=True)

class Route(Base):
    __tablename__ = "routes"

    train_number = Column(String, primary_key=True, index=True, nullable=False)
    sequence = Column(Integer, primary_key=True, nullable=False)
    station_code = Column(String, nullable=False)
    arrival_time = Column(String, nullable=True)
    departure_time = Column(String, nullable=True)
    distance_km = Column(Integer, nullable=True)

class Fare(Base):
    __tablename__ = "fares"

    train_number = Column(String, primary_key=True, nullable=False)
    class_type = Column(String, primary_key=True, nullable=False)   # e.g., SL, 3A
    amount = Column(Float, nullable=False)

class Booking(Base):
    __tablename__ = "bookings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    train_number = Column(String, nullable=False)
    passenger_count = Column(Integer, nullable=False)
    travel_class = Column(String, nullable=False)
    travel_date = Column(DateTime, nullable=False)
    status = Column(String, default="CONFIRMED")
    created_at = Column(DateTime, default=datetime.utcnow)