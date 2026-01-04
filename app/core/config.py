from functools import lru_cache
from pydantic import  Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Centralized application configuration (Pydantic v2).
    Loaded once at startup and reused across the app.
    """

    # -------------------------
    # Application
    # -------------------------
    APP_NAME: str = "Enterprise RAG Platform"
    ENV: str = Field(default="local", description="Environment name")

    # -------------------------
    # Pinecone Configuration
    # -------------------------
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "enterprise-rag-index"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    NAME_SPACE: str = "Enterprise-RAG"
    RAG_ACCESS_LEVELS : dict[str, int]= {
    "public": 1,
    "internal": 2,
    "sensitive": 3,
    "restrictive": 4,
    }


    # -------------------------
    # Embeddings Configuration
    # -------------------------
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # -------------------------
    # Chunking Configuration
    # -------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    # -------------------------
    # Runtime / ML Configuration
    # -------------------------
    TOKENIZERS_PARALLELISM: bool = False

    # -------------------------
    # Ingestion Configuration
    # -------------------------
    MAX_UPLOAD_MB: int = 20

    OPENAI_API_KEY: str

    
    # LLM Configuration
    NVIDIA_API_KEY: str
    NVIDIA_BASE_URL: str
    NVIDIA_MODELS: str
    NVIDIA_DEFAULT_MODEL: str



    # -------------------------
    # Validation (Pydantic v2)
    # -------------------------
    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        chunk_size = info.data.get("CHUNK_SIZE")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return v
    
    @property
    def nvidia_model_list(self) -> list[str]:
        return [m.strip() for m in self.NVIDIA_MODELS.split(",")]


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    Ensures config is loaded once per process.
    """
    return Settings()


# Singleton-style access
settings = get_settings()
