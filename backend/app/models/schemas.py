from datetime import datetime
from typing import Any, Dict, List, Optional
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

class BookingDraft(BaseModel):
    source: Optional[str] = None
    destination: Optional[str] = None
    travel_date: Optional[str] = None
    travel_class: Optional[str] = None
    passenger_count: Optional[int] = None
    train_number: Optional[str] = None
    time_preference: Optional[str] = None
    departure_after: Optional[str] = None
    departure_before: Optional[str] = None
    berth_preference: Optional[str] = None
    budget: Optional[int] = None
    direct_only: bool = False
    ready_for_submit: bool = False
    missing_required_fields: List[str] = Field(default_factory=list)

class ChatDiagnostics(BaseModel):
    intent: Optional[str] = None
    route: str = "unknown"
    llm_attempted: bool = False
    llm_used: bool = False
    local_handler_used: bool = False
    fallback_used: bool = False
    llm_error: Optional[str] = None
    local_error: Optional[str] = None

class StructuredChatResponse(BaseModel):
    messages: List[ChatMessage]
    action: Optional[str] = None
    booking_draft: Optional[BookingDraft] = None
    missing_required_fields: List[str] = Field(default_factory=list)
    diagnostics: ChatDiagnostics = Field(default_factory=ChatDiagnostics)
    metadata: Dict[str, Any] = Field(default_factory=dict)
