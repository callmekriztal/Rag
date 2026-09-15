import os
from functools import lru_cache
from typing import Set
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = ""
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash"
    PERSIST_DIR: str = "faiss_store"
    DATA_DIR: str = "data"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_UPLOAD_SIZE: int = 20 * 1024 * 1024  # 20 MB
    ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".txt", ".docx", ".csv", ".json", ".xlsx"}

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
