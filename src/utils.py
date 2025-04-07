"""
Utility functions for the Aetna PDF Chatbot.
"""
import os
import logging
from typing import Optional

from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment() -> None:
    """
    Check and validate environment setup.
    """
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY environment variable not set")
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    
    # Create necessary directories
    os.makedirs("./data/pdf", exist_ok=True)
    os.makedirs("./data/vector_db", exist_ok=True)
    
    logger.info("Environment setup validated")

def initialize_vector_store(force_rebuild: bool = False) -> VectorStore:
    """
    Initialize the vector store, processing the PDF if necessary.
    
    Args:
        force_rebuild: Whether to force rebuilding the vector store
        
    Returns:
        Initialized VectorStore
    """
    # Initialize vector store
    vector_store = VectorStore()
    
    # Check if vector store exists
    if force_rebuild or not os.path.exists(vector_store.vector_db_path) or not os.listdir(vector_store.vector_db_path):
        logger.info("Vector store not found or rebuild forced, processing PDF...")
        
        # Process PDF
        pdf_processor = PDFProcessor()
        chunks = pdf_processor.process_pdf()
        
        # Create vector store
        vector_store.create_vector_store(chunks)
    else:
        # Load existing vector store
        logger.info("Loading existing vector store...")
        vector_store.load_vector_store()
    
    return vector_store
