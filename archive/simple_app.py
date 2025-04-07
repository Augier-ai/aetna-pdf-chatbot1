"""
Simple Aetna PDF Chatbot Application

This is a simplified version of the chatbot that uses the OpenAI API directly
to avoid issues with project-scoped API keys.
"""
import os
import sys
import logging
import requests
import PyPDF2
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

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

# Load environment variables
load_dotenv()

# Constants
PDF_URL = "https://www.aetnabetterhealth.com/content/dam/aetna/medicaid/illinois/pdf/ABHIL_Member_Handbook.pdf"
PDF_LOCAL_PATH = "./data/pdf/ABHIL_Member_Handbook.pdf"
APP_TITLE = "Aetna Better Health Illinois Member Handbook Chatbot"
APP_DESCRIPTION = "Ask questions about your Aetna Better Health Illinois benefits and coverage."
DEFAULT_GREETING = "Hello! I'm your Aetna Better Health Illinois assistant. How can I help you today?"

class SimplePDFChatbot:
    """
    A simplified PDF chatbot that uses OpenAI API directly.
    """
    
    def __init__(self):
        """Initialize the chatbot."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.pdf_content = None
        self.conversation_history = []
        
        # Ensure PDF is downloaded and processed
        self.download_and_process_pdf()
    
    def download_and_process_pdf(self):
        """Download and process the PDF."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(PDF_LOCAL_PATH), exist_ok=True)
        
        # Download PDF if it doesn't exist
        if not os.path.exists(PDF_LOCAL_PATH):
            logger.info(f"Downloading PDF from {PDF_URL}")
            try:
                response = requests.get(PDF_URL, stream=True)
                response.raise_for_status()
                
                with open(PDF_LOCAL_PATH, 'wb') as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)
                
                logger.info(f"PDF downloaded to {PDF_LOCAL_PATH}")
            except Exception as e:
                logger.error(f"Error downloading PDF: {str(e)}")
                raise
        else:
            logger.info(f"PDF already exists at {PDF_LOCAL_PATH}")
        
        # Extract text from PDF
        self.pdf_content = self.extract_text_from_pdf()
        logger.info("PDF processed successfully")
    
    def extract_text_from_pdf(self):
        """Extract text from the PDF."""
        try:
            text = ""
            with open(PDF_LOCAL_PATH, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += f"Page {page_num + 1}:\n{page_text}\n\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def get_response(self, query):
        """Get a response from the OpenAI API."""
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Create system message with context about the PDF
        system_message = {
            "role": "system", 
            "content": f"""You are an AI assistant for Aetna Better Health Illinois. 
            You provide helpful, accurate, and friendly information about the Aetna Better Health Illinois Member Handbook. 
            Your goal is to assist members in understanding their benefits, coverage, and how to navigate their healthcare.
            
            Here is the content of the PDF that you can reference to answer questions:
            
            {self.pdf_content[:10000]}  # First 10000 characters of the PDF
            
            Always maintain a helpful and professional tone. If you don't know the answer, just say that you don't know,
            don't try to make up an answer."""
        }
        
        try:
            # Get response from OpenAI
            messages = [system_message] + self.conversation_history
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                max_tokens=800
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            logger.error(f"Error getting response from OpenAI: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your request. Please try again later. Error: {str(e)}"
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")

def create_interface(chatbot):
    """Create the Gradio interface."""
    with gr.Blocks(title=APP_TITLE) as interface:
        # Header
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)
        
        # Chat interface
        chatbot_ui = gr.Chatbot(
            height=500,
            bubble_full_width=False,
            show_copy_button=True,
            avatar_images=("👤", "🏥"),
        )
        
        # Input components
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask a question about your Aetna Better Health Illinois benefits...",
                scale=9,
                container=False,
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")
        
        # Clear button
        clear_btn = gr.Button("Clear Conversation")
        
        # Add system message on page load
        gr.on(
            triggers=[gr.load],
            fn=lambda: [[None, DEFAULT_GREETING]],
            outputs=[chatbot_ui],
        )
        
        # Handle message submission
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, ""
            
            # Add user message to chat history
            chat_history.append([message, None])
            
            # Get response from chatbot
            response = chatbot.get_response(message)
            
            # Update chat history with response
            chat_history[-1][1] = response
            
            return chat_history, ""
        
        # Connect event handlers
        submit_btn.click(
            respond,
            inputs=[msg, chatbot_ui],
            outputs=[chatbot_ui, msg],
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot_ui],
            outputs=[chatbot_ui, msg],
        )
        
        # Clear conversation
        def clear_conversation():
            chatbot.reset_conversation()
            return [[None, DEFAULT_GREETING]]
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot_ui],
        )
    
    return interface

def main():
    """Main application entry point."""
    try:
        # Initialize chatbot
        logger.info("Initializing chatbot...")
        chatbot = SimplePDFChatbot()
        
        # Create and launch interface
        logger.info("Creating interface...")
        interface = create_interface(chatbot)
        
        logger.info("Launching interface...")
        interface.launch(share=False)
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        raise

if __name__ == "__main__":
    main()
