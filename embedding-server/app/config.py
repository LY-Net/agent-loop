
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # Application
    APP_NAME: str = "Local Embedding Server"
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8090
    APP_DEBUG: bool = False

    # Model
    EMBEDDING_MODEL_PATH: str = r"G:\models\bge-m3"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_USE_FP16: bool = False

    # Inference
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 8
    EMBEDDING_MAX_LENGTH: int = 512
    EMBEDDING_CPU_THREADS: int = 6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
