"""
PDF processing utilities for extracting and processing text from the Aetna PDF.
"""
import os
import logging
import requests
import PyPDF2
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.config import PDF_URL, PDF_LOCAL_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF document processing, including downloading, text extraction, and chunking.
    """
    
    def __init__(self, pdf_url: str = PDF_URL, pdf_path: str = PDF_LOCAL_PATH):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_url: URL to download the PDF from
            pdf_path: Local path to save/load the PDF
        """
        self.pdf_url = pdf_url
        self.pdf_path = pdf_path
        
    def download_pdf(self) -> str:
        """
        Download the PDF if it doesn't exist locally.
        
        Returns:
            Path to the local PDF file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.pdf_path), exist_ok=True)
        
        if not os.path.exists(self.pdf_path):
            logger.info(f"Downloading PDF from {self.pdf_url}")
            try:
                response = requests.get(self.pdf_url, stream=True)
                response.raise_for_status()
                
                with open(self.pdf_path, 'wb') as pdf_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        pdf_file.write(chunk)
                
                logger.info(f"PDF downloaded to {self.pdf_path}")
            except Exception as e:
                logger.error(f"Error downloading PDF: {str(e)}")
                raise
        else:
            logger.info(f"PDF already exists at {self.pdf_path}")
            
        return self.pdf_path
    
    def extract_text(self) -> str:
        """
        Extract all text from the PDF.
        
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(self.pdf_path):
            self.download_pdf()
            
        try:
            text = ""
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += f"Page {page_num + 1}:\n{page_text}\n\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, 
                  chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
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
        pages = text.split("Page ")
        if pages and not pages[0].strip():  # Remove empty first element if it exists
            pages = pages[1:]
            
        for page_content in pages:
            if not page_content.strip():
                continue
                
            # Get page number
            page_lines = page_content.split("\n", 1)
            if len(page_lines) < 2:
                continue
                
            try:
                page_num = int(page_lines[0].strip().rstrip(':'))
                content = page_lines[1].strip()
            except (ValueError, IndexError):
                continue
                
            # Further split page content into paragraphs
            paragraphs = content.split("\n\n")
            
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
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self) -> List[Dict[str, Any]]:
        """
        Process the PDF: download, extract text, and chunk it.
        
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Download PDF if needed
        if not os.path.exists(self.pdf_path):
            self.download_pdf()
        
        # Extract text
        text = self.extract_text()
        
        # Chunk text
        chunks = self.chunk_text(text)
        
        return chunks
