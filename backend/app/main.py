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

@app.get("/", tags=["Health"])
def root():
    """Root endpoint for Render health check and API overview."""
    return {
        "message": "RailMitra API is running",
        "docs": "/docs",
        "health": "/health",
        "api_version": "/api/v1"
    }

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
