import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

if os.path.exists('./docker/.env'):
    load_dotenv('./docker/.env')
else:
    load_dotenv()


class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "RAGAS Evaluation Service"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database


    POSTGRES_USER:str="postgresql"
    POSTGRES_PASSWORD:str="password"
    POSTGRES_HOSTNAME:str="localhost"
    POSTGRES_DATABASE:str="ragas"
    POSTGRES_PORT:int=5432

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOSTNAME}:{self.POSTGRES_PORT}/{self.POSTGRES_DATABASE}"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # LLM Providers
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"

    ANTHROPIC_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None

    # LANGSMITH
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_TRACING: Optional[bool] = None
    LANGSMITH_ENDPOINT: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Evaluation Settings
    DEFAULT_BATCH_SIZE: int = 10
    MAX_BATCH_SIZE: int = 100
    EVALUATION_TIMEOUT: int = 3600  # 1 hour
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = "./docker/.env"
        case_sensitive = True


settings = Settings()
