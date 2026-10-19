from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "Production RAG & Evaluation Harness"
    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Retrieval Hyperparameters
    DENSE_TOP_K: int = 15
    SPARSE_TOP_K: int = 15
    RERANK_TOP_K: int = 5
    RRF_K: int = 60  # Standard constant for Reciprocal Rank Fusion
    
    # Chunking Hyperparameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 75
    
    # Storage settings
    VECTOR_BACKEND: str = "inmemory"  # "inmemory" or "qdrant"
    QDRANT_HOST: Optional[str] = "localhost"
    QDRANT_PORT: int = 6333
    
    # Optional LLM API settings
    OPENAI_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
