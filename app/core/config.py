import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    APP_ENV: str = Field(os.getenv("APP_ENV", "development"), env="APP_ENV")
    DEBUG: bool = Field(os.getenv("DEBUG", "True").lower() == "true", env="DEBUG")
    LOG_LEVEL: str = Field(os.getenv("LOG_LEVEL", "INFO"), env="LOG_LEVEL")
    
    # API settings
    API_HOST: str = Field(os.getenv("API_HOST", "0.0.0.0"), env="API_HOST")
    API_PORT: int = Field(int(os.getenv("API_PORT", "8000")), env="API_PORT")
    
    # OpenAI settings
    OPENAI_API_KEY: str = Field(os.getenv("OPENAI_API_KEY", ""), env="OPENAI_API_KEY")
    
    # PDF settings
    PDF_URL: str = Field(
        os.getenv(
            "PDF_URL", 
            "https://www.aetnabetterhealth.com/content/dam/aetna/medicaid/illinois/pdf/ABHIL_Member_Handbook.pdf"
        ), 
        env="PDF_URL"
    )
    PDF_LOCAL_PATH: str = Field(
        os.getenv("PDF_LOCAL_PATH", "./data/pdf/ABHIL_Member_Handbook.pdf"), 
        env="PDF_LOCAL_PATH"
    )
    
    # Vector DB settings
    VECTOR_DB_PATH: str = Field(os.getenv("VECTOR_DB_PATH", "./data/vectordb"), env="VECTOR_DB_PATH")
    EMBEDDING_MODEL: str = Field(os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"), env="EMBEDDING_MODEL")
    
    # LLM settings
    LLM_MODEL: str = Field(os.getenv("LLM_MODEL", "gpt-4"), env="LLM_MODEL")
    LLM_TEMPERATURE: float = Field(float(os.getenv("LLM_TEMPERATURE", "0.3")), env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(int(os.getenv("LLM_MAX_TOKENS", "800")), env="LLM_MAX_TOKENS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()
