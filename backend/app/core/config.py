import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_db_url():
    if os.path.exists("/app/train_ticket.db"):
        return "sqlite:////app/train_ticket.db"
    if os.path.exists("train_ticket.db"):
        return "sqlite:///train_ticket.db"
    return "sqlite:///C:/fare/train-ticket-assistant/scripts/train_ticket.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",   # ignore Render's extra env vars like PORT
    )

    SERVER_HOST: str = Field(default="0.0.0.0")
    SERVER_PORT: int = Field(default=8000)
    PORT: int | None = Field(default=None)   # optional, Render injects this
    DEBUG: bool = Field(default=False)

    DATABASE_URL: str = Field(default_factory=get_default_db_url)
    JWT_SECRET_KEY: str = Field(default="super-secret-jwt-key")
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60)

    # Hugging Face – try both common env var names
    HUGGINGFACEHUB_API_TOKEN: str = Field(default="")
    HUGGINGFACE_API_KEY: str = Field(default="")

    @property
    def hf_token(self) -> str:
        """Return whichever HF token variable is set."""
        return self.HUGGINGFACEHUB_API_TOKEN or self.HUGGINGFACE_API_KEY


settings = Settings()
