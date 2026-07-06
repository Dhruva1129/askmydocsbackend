# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # App
    APP_NAME: str = "AskMyDocs"
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173"]
    JWT_SECRET_KEY: str = "super-secret-jwt-key-for-development-change-in-production"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/askmydocs"

    # Vector DB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "askmydocs"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # free, runs locally

    # LLM  (Groq — free tier)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Retrieval
    BM25_TOP_K: int = 20
    VECTOR_TOP_K: int = 20
    RERANK_TOP_K: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Eval thresholds (RAGAS)
    MIN_FAITHFULNESS: float = 0.80
    MIN_ANSWER_RELEVANCY: float = 0.75
    MIN_CONTEXT_RECALL: float = 0.70

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
