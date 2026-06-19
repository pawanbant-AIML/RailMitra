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
    title="Rail Mitra",
    description="AI-powered Indian Railways assistant — natural language train search, booking, fares and more.",
    version="1.1.0",
)

# ---------------------------------------------------------------------------
# CORS
# Use an explicit origin list. Wildcard + allow_credentials is a browser spec
# violation and modern browsers reject it. Read from env so production can
# lock this down without a code change.
# ---------------------------------------------------------------------------
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,https://railmitra-frontend.onrender.com",
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(trains.router, prefix="/api/v1", tags=["Trains"])
app.include_router(stations.router, prefix="/api/v1", tags=["Stations"])
app.include_router(routes.router, prefix="/api/v1", tags=["Routes"])
app.include_router(schedules.router, prefix="/api/v1", tags=["Schedules"])
app.include_router(fares.router, prefix="/api/v1", tags=["Fares"])
app.include_router(bookings.router, prefix="/api/v1", tags=["Bookings"])


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Rail Mitra API starting up. Allowed origins: %s", ALLOWED_ORIGINS)


@app.get("/", tags=["Health"])
def root():
    """Root health / discovery endpoint."""
    return {
        "app": "Rail Mitra",
        "version": "1.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
    }


@app.get("/health", tags=["Health"])
def health_check():
    logger.info("Health check requested")
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.SERVER_PORT))
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=port,
        reload=settings.DEBUG,
    )
