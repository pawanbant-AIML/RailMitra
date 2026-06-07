#!/usr/bin/env python
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import (
    chat,
    trains,
    stations,
    routes,
    schedules,
    fares,
    bookings,
)
from app.core.config import settings
from app.core.logger import logger

app = FastAPI(
    title="AI Train Ticket Assistant",
    description="Natural‑language railway assistant for Indian Railways.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(trains.router, prefix="/api/v1", tags=["Trains"])
app.include_router(stations.router, prefix="/api/v1", tags=["Stations"])
app.include_router(routes.router, prefix="/api/v1", tags=["Routes"])
app.include_router(schedules.router, prefix="/api/v1", tags=["Schedules"])
app.include_router(fares.router, prefix="/api/v1", tags=["Fares"])
app.include_router(bookings.router, prefix="/api/v1", tags=["Bookings"])

@app.get("/health", tags=["Health"])
def health_check():
    logger.info("Health check requested")
    return {"status": "ok"}

if __name__ == "__main__":
    # Use PORT environment variable if available (Render), otherwise fallback to settings
    port = int(os.environ.get("PORT", settings.SERVER_PORT))
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=port,
        reload=settings.DEBUG,
    )

backend/app/core/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=True)

    SERVER_HOST: str = Field(default="0.0.0.0")
    SERVER_PORT: str = Field(default="8000")
    DEBUG: bool = Field(default=True)

    DATABASE_URL: str = Field(
        default="sqlite:///C:/fare/train-ticket-assistant/scripts/train_ticket.db"
    )
    JWT_SECRET_KEY: str = Field(default="super-secret-jwt-key")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

settings = Settings()
