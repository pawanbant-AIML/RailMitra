from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class TrainBase(BaseModel):
    train_number: str
    train_name: str
    source_station_code: str
    destination_station_code: str

class TrainCreate(TrainBase):
    pass

class Train(TrainBase):
    class Config:
        from_attributes = True

class StationBase(BaseModel):
    station_code: str
    station_name: str
    city: Optional[str]

class Station(StationBase):
    class Config:
        from_attributes = True

class RouteBase(BaseModel):
    train_number: str
    sequence: int
    station_code: str
    arrival_time: Optional[str]
    departure_time: Optional[str]
    distance_km: Optional[int]

class Route(RouteBase):
    class Config:
        from_attributes = True

class FareBase(BaseModel):
    train_number: str
    class_type: str
    amount: float

class Fare(FareBase):
    class Config:
        from_attributes = True

class BookingBase(BaseModel):
    user_id: int
    train_number: str
    passenger_count: int
    travel_class: str = Field(..., description="e.g., SL")
    travel_date: datetime

class BookingCreate(BookingBase):
    pass

class Booking(BookingBase):
    id: int
    status: str = "CONFIRMED"
    created_at: datetime
    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    role: str
    content: str