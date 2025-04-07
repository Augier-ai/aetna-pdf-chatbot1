"""
Configuration settings for the Aetna PDF Chatbot.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# PDF settings
PDF_URL = os.getenv(
    "PDF_URL", 
    "https://www.aetnabetterhealth.com/content/dam/aetna/medicaid/illinois/pdf/ABHIL_Member_Handbook.pdf"
)
PDF_LOCAL_PATH = os.getenv("PDF_LOCAL_PATH", "./data/pdf/ABHIL_Member_Handbook.pdf")

# Vector DB settings
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Application settings
APP_TITLE = "Aetna Better Health Illinois Member Handbook Chatbot"
APP_DESCRIPTION = "Ask questions about your Aetna Better Health Illinois benefits and coverage."
DEFAULT_GREETING = "Hello! I'm your Aetna Better Health Illinois assistant. How can I help you today?"
