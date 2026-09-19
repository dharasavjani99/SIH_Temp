"""Settings are read from the environment only. No secret is ever hardcoded
or sent to the frontend."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "ResQTech API"
    version: str = "0.9.2"

    database_url: str = "sqlite:///./resqtech.db"

    jwt_secret: str = "dev-only-insecure-secret"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 480

    cors_origins: str = "http://localhost:5173"

    data_mode: str = "demo"          # "demo" | "live"
    imd_api_key: str = ""
    cwc_api_key: str = ""
    bhuvan_api_key: str = ""

    llm_provider: str = "anthropic"
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    risk_model_path: str = "app/ml/artifacts/risk_model.joblib"
    seg_model_path: str = "app/dl/artifacts/impact_seg.pt"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
