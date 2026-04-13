from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "My FastAPI App"
    app_env: str = "development"
    app_debug: bool = True

    api_prefix: str = "/api"

    # ตอน deploy ค่อยเปลี่ยนเป็น domain จริงของ PHP
    allowed_origins: str = "http://localhost,http://127.0.0.1,http://localhost:80"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def cors_origins_list(self) -> list[str]:
        return [x.strip() for x in self.allowed_origins.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()