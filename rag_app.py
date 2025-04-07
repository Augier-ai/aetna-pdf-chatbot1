"""
Aetna PDF Chatbot with RAG (Retrieval-Augmented Generation)

This version implements proper text chunking and vector search using ChromaDB
for improved performance and accuracy.
"""
import os
import logging
import requests
import PyPDF2
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import re
import time
import chromadb
from chromadb.utils import embedding_functions

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
VECTOR_DB_PATH = "./data/vector_db"
APP_TITLE = "Aetna Better Health Illinois Member Handbook Chatbot (RAG Enhanced)"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

# Ensure directories exist
os.makedirs(os.path.dirname(PDF_LOCAL_PATH), exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def clean_vector_db_directory():
    """
    Clean the vector database directory to ensure a fresh start.
    """
    import shutil
    if os.path.exists(VECTOR_DB_PATH):
        try:
            shutil.rmtree(VECTOR_DB_PATH)
            logger.info(f"Removed existing vector database directory: {VECTOR_DB_PATH}")
        except Exception as e:
            logger.error(f"Error removing vector database directory: {str(e)}")
    
    # Recreate the directory
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

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

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split the text into overlapping chunks for processing.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries containing chunk text and metadata
    """
    chunks = []
    
    # Split text by page markers
    pages = re.split(r'Page (\d+):', text)
    
    # Process the split result
    for i in range(1, len(pages), 2):
        if i < len(pages):
            try:
                page_num = int(pages[i])
                page_content = pages[i+1] if i+1 < len(pages) else ""
                
                if not page_content.strip():
                    continue
                
                # Further split page content into paragraphs
                paragraphs = re.split(r'\n\s*\n', page_content)
                
                current_chunk = ""
                current_chunk_metadata = {
                    "page": page_num,
                    "source": f"Page {page_num} of Aetna Better Health Illinois Member Handbook"
                }
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    
                    # If adding this paragraph would exceed chunk size, save current chunk and start a new one
                    if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "metadata": current_chunk_metadata.copy()
                        })
                        
                        # Start new chunk with overlap
                        if len(current_chunk) > chunk_overlap:
                            overlap_text = current_chunk[-chunk_overlap:]
                            current_chunk = overlap_text + "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                    else:
                        # Add paragraph to current chunk
                        if current_chunk:
                            current_chunk += "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                
                # Add the last chunk from this page
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": current_chunk_metadata.copy()
                    })
            except ValueError:
                continue
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def create_vector_db(chunks):
    """
    Create a vector database from text chunks.
    
    Args:
        chunks: List of dictionaries containing text and metadata
        
    Returns:
        ChromaDB collection
    """
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    
    # Use OpenAI embeddings
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-ada-002"
    )
    
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(name="aetna_handbook")
        logger.info("Deleted existing collection")
    except Exception:
        pass
        
    # Create new collection
    collection = chroma_client.create_collection(name="aetna_handbook", embedding_function=openai_ef)
    logger.info("Created new vector database collection")
    
    # Check if collection is empty
    if collection.count() == 0:
        logger.info("Adding documents to vector database...")
        
        # Prepare data for batch addition
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add documents in batches to avoid timeouts
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            logger.info(f"Added batch {i//batch_size + 1} to vector database")
            time.sleep(1)  # Small delay to avoid rate limits
    
    return collection

def query_vector_db(collection, query, n_results=5):
    """
    Query the vector database for relevant chunks.
    
    Args:
        collection: ChromaDB collection
        query: Query string
        n_results: Number of results to return
        
    Returns:
        List of relevant text chunks
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Extract documents and metadata
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        # Combine documents with their metadata
        context_chunks = []
        for doc, meta in zip(documents, metadatas):
            context_chunks.append(f"Source: {meta.get('source', 'Unknown')}\n\n{doc}")
        
        return context_chunks
    except Exception as e:
        logger.error(f"Error querying vector database: {str(e)}")
        return []

# Global variables for caching
pdf_text = None
collection = None
conversation_history = []

def initialize_database():
    """Initialize the vector database once at startup."""
    global pdf_text, collection
    
    # Download PDF if needed
    if not os.path.exists(PDF_LOCAL_PATH):
        download_pdf()
    
    # Extract text from PDF
    pdf_text = extract_text_from_pdf()
    logger.info("PDF text extracted")
    
    # Chunk text
    chunks = chunk_text(pdf_text)
    logger.info(f"Text split into {len(chunks)} chunks")
    
    # Create vector database
    collection = create_vector_db(chunks)
    logger.info("Vector database created and ready for queries")
    
    return collection

def chat(message, history):
    """Process a chat message and return a response."""
    global conversation_history, collection
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": message})
    
    try:
        # Ensure collection is initialized
        if collection is None:
            logger.info("Vector database not initialized, initializing now...")
            collection = initialize_database()
        
        # Query vector database for relevant chunks
        context_chunks = query_vector_db(collection, message)
        
        # Combine chunks into context
        context = "\n\n".join(context_chunks)
        
        # Create system message with context about the PDF
        system_message = {
            "role": "system", 
            "content": f"""You are an AI assistant for Aetna Better Health Illinois. 
            You provide helpful, accurate, and friendly information about the Aetna Better Health Illinois Member Handbook. 
            Your goal is to assist members in understanding their benefits, coverage, and how to navigate their healthcare.
            
            Use ONLY the following information from the handbook to answer the user's question:
            
            {context}
            
            Always maintain a helpful and professional tone. If you don't know the answer or if the information is not provided
            in the context, just say that you don't know or that the information is not available in the handbook.
            Do not make up information."""
        }
        
        # Get response from OpenAI
        messages = [system_message] + conversation_history[-5:]  # Only use last 5 messages for context window management
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,  # Lower temperature for more factual responses
            max_tokens=600
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        logger.error(f"Error getting response: {str(e)}")
        return f"I'm sorry, I encountered an error while processing your request. Please try again later. Error: {str(e)}"

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title=APP_TITLE,
    description="Ask questions about your Aetna Better Health Illinois benefits and coverage. This version uses RAG (Retrieval-Augmented Generation) for improved performance and accuracy.",
    examples=[
        "What healthcare services are covered?",
        "How do I find a doctor?",
        "What are my dental benefits?",
        "How do I file a complaint?",
        "What is the nurse advice line?"
    ],
    theme="soft"
)

# Clean vector database directory for a fresh start
clean_vector_db_directory()

# Ensure PDF is downloaded
download_pdf()

# Initialize the database once at startup
logger.info("Initializing vector database at startup...")
initialize_database()
logger.info("Database initialization complete")

# For Vercel deployment - expose the app
app = demo.app

# For local development
if __name__ == "__main__":
    # Launch the interface
    demo.launch()
