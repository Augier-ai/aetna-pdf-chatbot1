import os
import logging
import fitz  # PyMuPDF
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from ..core.config import settings

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Service for processing PDF documents, extracting text, and preparing it for embedding
    """
    
    def __init__(self, pdf_url: Optional[str] = None, pdf_path: Optional[str] = None):
        """
        Initialize the PDF processor
        
        Args:
            pdf_url: URL to download the PDF from
            pdf_path: Local path to the PDF file
        """
        self.pdf_url = pdf_url or settings.PDF_URL
        self.pdf_path = pdf_path or settings.PDF_LOCAL_PATH
        
    def download_pdf(self) -> str:
        """
        Download the PDF from the URL if it doesn't exist locally
        
        Returns:
            Path to the downloaded PDF file
        """
        pdf_dir = os.path.dirname(self.pdf_path)
        os.makedirs(pdf_dir, exist_ok=True)
        
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
        Extract all text from the PDF
        
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(self.pdf_path):
            self.download_pdf()
            
        try:
            text = ""
            with fitz.open(self.pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text += f"Page {page_num + 1}:\n{page.get_text()}\n\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def extract_text_with_structure(self) -> List[Dict[str, Any]]:
        """
        Extract text from the PDF with structural information
        
        Returns:
            List of dictionaries containing page number, text, and metadata
        """
        if not os.path.exists(self.pdf_path):
            self.download_pdf()
            
        try:
            structured_text = []
            with fitz.open(self.pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    # Extract text
                    text = page.get_text()
                    
                    # Extract metadata
                    blocks = page.get_text("dict")["blocks"]
                    
                    structured_text.append({
                        "page_num": page_num + 1,
                        "text": text,
                        "blocks": blocks
                    })
            
            return structured_text
        except Exception as e:
            logger.error(f"Error extracting structured text from PDF: {str(e)}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split the text into overlapping chunks for processing
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        chunks = []
        
        # Split text by page markers
        pages = re.split(r'Page \d+:', text)
        if pages and not pages[0].strip():  # Remove empty first element if it exists
            pages = pages[1:]
            
        current_page = 1
        
        for page_content in pages:
            page_content = page_content.strip()
            if not page_content:
                current_page += 1
                continue
                
            # Further split page content into paragraphs
            paragraphs = re.split(r'\n\s*\n', page_content)
            
            current_chunk = ""
            current_chunk_metadata = {
                "start_page": current_page,
                "end_page": current_page
            }
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                # If adding this paragraph would exceed chunk size, save current chunk and start a new one
                if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": current_chunk_metadata
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + paragraph
                    current_chunk_metadata = {
                        "start_page": current_page,
                        "end_page": current_page
                    }
                else:
                    # Add paragraph to current chunk
                    current_chunk += " " + paragraph
            
            # Add the last chunk from this page
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": current_chunk_metadata
                })
                
            current_page += 1
        
        return chunks
    
    def process_pdf(self, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Process the PDF: download, extract text, and chunk it
        
        Args:
            chunk_size: Maximum size of each text chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        # Download PDF if needed
        if not os.path.exists(self.pdf_path):
            self.download_pdf()
        
        # Extract text
        text = self.extract_text()
        
        # Chunk text
        chunks = self.chunk_text(text, chunk_size, overlap)
        
        logger.info(f"Processed PDF into {len(chunks)} chunks")
        
        return chunks
