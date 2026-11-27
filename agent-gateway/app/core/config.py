
from functools import lru_cache
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # Application
    APP_NAME: str = "Agent Gateway"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8080

    # DeepSeek Model
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-flash"

    # LLM Request Settings
    DEEPSEEK_TIMEOUT: float = 30.0
    DEEPSEEK_MAX_RETRIES: int = 2

    # Embedding
    EMBEDDING_PROVIDER: str = "local_bge"  # 选择 Embedding Provide

    EMBEDDING_BASE_URL: str = "http://127.0.0.1:8090" #	Embedding Server 地址
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3" # 模型名称
    EMBEDDING_DIMENSION: int = 1024 # 向量维度

    EMBEDDING_TIMEOUT: float = 60.0 # HTTP 请求超时时间
    EMBEDDING_BATCH_SIZE: int = 8 # 模型推理批次大小
    EMBEDDING_MAX_TEXTS: int = 100 # 单次请求最大文本数量
    EMBEDDING_MAX_TEXT_LENGTH: int = 4000 #	单条文本最大字符数

    # PostgreSQL
    DATABASE_NAME: str = "agent_gateway"
    DATABASE_PASSWORD: str = ""
    DATABASE_HOST: str = "127.0.0.1"
    DATABASE_USER: str = "postgres"
    DATABASE_PORT: int = 5432

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    @property
    def database_url(self) -> str:
        password = quote_plus(self.DATABASE_PASSWORD)

        return (
            f"postgresql+asyncpg://"
            f"{self.DATABASE_USER}:{password}@"
            f"{self.DATABASE_HOST}:{self.DATABASE_PORT}/"
            f"{self.DATABASE_NAME}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
