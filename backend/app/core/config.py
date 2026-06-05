import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

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