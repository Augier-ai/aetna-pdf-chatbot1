"""
Main application entry point for the Aetna PDF Chatbot.
"""
import os
import logging
import argparse
from dotenv import load_dotenv

from src.utils import setup_environment, initialize_vector_store
from src.llm_service import LLMService
from src.chat_interface import ChatInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main application entry point.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Aetna PDF Chatbot")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of vector store")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on")
    parser.add_argument("--share", action="store_true", help="Create a public link for the app")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Setup environment
        setup_environment()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = initialize_vector_store(force_rebuild=args.rebuild)
        
        # Initialize LLM service
        logger.info("Initializing LLM service...")
        llm_service = LLMService(vector_store)
        
        # Create and launch chat interface
        logger.info("Creating chat interface...")
        chat_interface = ChatInterface(llm_service)
        
        logger.info(f"Launching app on port {args.port}...")
        chat_interface.launch(server_port=args.port, share=args.share)
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        raise

if __name__ == "__main__":
    main()
