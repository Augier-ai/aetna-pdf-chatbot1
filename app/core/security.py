import os
import logging
from typing import Dict, Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from .config import settings

logger = logging.getLogger(__name__)

# Optional API key security for production environments
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)) -> Optional[str]:
    """
    Validate API key if it's configured in the environment
    """
    # If no API key is set in environment, skip validation (for development)
    if not settings.APP_ENV == "production":
        return None
    
    # In production, require API key
    if api_key_header is None:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="API key is required"
        )
    
    # Check if the provided API key is valid
    expected_api_key = os.getenv("API_KEY")
    if expected_api_key and api_key_header != expected_api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    
    return api_key_header

def validate_openai_api_key() -> None:
    """
    Validate that OpenAI API key is configured
    """
    if not settings.OPENAI_API_KEY:
        logger.error("OpenAI API key is not configured")
        raise ValueError("OpenAI API key is not configured. Please set OPENAI_API_KEY environment variable.")
