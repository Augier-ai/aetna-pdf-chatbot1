"""
Basic Aetna PDF Chatbot Application

This is a basic version of the chatbot that uses the OpenAI API directly
with a simple Gradio interface.
"""
import os
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
PDF_URL = "https://www.aetnabetterhealth.com/content/dam/aetna/medicaid/illinois/pdf/ABHIL_Member_Handbook.pdf"
PDF_LOCAL_PATH = "./data/pdf/ABHIL_Member_Handbook.pdf"
APP_TITLE = "Aetna Better Health Illinois Member Handbook Chatbot"

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

# Ensure PDF directory exists
os.makedirs(os.path.dirname(PDF_LOCAL_PATH), exist_ok=True)

def download_pdf():
    """Download the PDF if it doesn't exist locally."""
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
    
    return PDF_LOCAL_PATH

def extract_text_from_pdf():
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

# Download PDF and extract text
download_pdf()
pdf_content = extract_text_from_pdf()
logger.info("PDF processed successfully")

# Store conversation history
conversation_history = []

def chat(message, history):
    """Process a chat message and return a response."""
    global conversation_history
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": message})
    
    # Create system message with context about the PDF
    system_message = {
        "role": "system", 
        "content": f"""You are an AI assistant for Aetna Better Health Illinois. 
        You provide helpful, accurate, and friendly information about the Aetna Better Health Illinois Member Handbook. 
        Your goal is to assist members in understanding their benefits, coverage, and how to navigate their healthcare.
        
        Here is the content of the PDF that you can reference to answer questions:
        
        {pdf_content[:10000]}  # First 10000 characters of the PDF
        
        Always maintain a helpful and professional tone. If you don't know the answer, just say that you don't know,
        don't try to make up an answer."""
    }
    
    try:
        # Get response from OpenAI
        messages = [system_message] + conversation_history
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        logger.error(f"Error getting response from OpenAI: {str(e)}")
        return f"I'm sorry, I encountered an error while processing your request. Please try again later. Error: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title=APP_TITLE,
    description="Ask questions about your Aetna Better Health Illinois benefits and coverage.",
    examples=[
        "What healthcare services are covered?",
        "How do I find a doctor?",
        "What are my dental benefits?",
        "How do I file a complaint?",
        "What is the nurse advice line?"
    ],
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
