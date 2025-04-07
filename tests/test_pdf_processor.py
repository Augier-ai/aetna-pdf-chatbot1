"""
Tests for the PDF processor module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import tempfile

from src.pdf_processor import PDFProcessor

class TestPDFProcessor:
    """
    Test cases for the PDFProcessor class.
    """
    
    def test_init(self):
        """Test initialization with default values."""
        processor = PDFProcessor()
        assert processor.pdf_url.endswith("ABHIL_Member_Handbook.pdf")
        assert processor.pdf_path.endswith("ABHIL_Member_Handbook.pdf")
    
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        custom_url = "https://example.com/test.pdf"
        custom_path = "/tmp/test.pdf"
        processor = PDFProcessor(pdf_url=custom_url, pdf_path=custom_path)
        assert processor.pdf_url == custom_url
        assert processor.pdf_path == custom_path
    
    @patch('requests.get')
    def test_download_pdf(self, mock_get):
        """Test downloading a PDF."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"test content"]
        mock_get.return_value = mock_response
        
        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Test download
            processor = PDFProcessor(pdf_path=temp_path)
            result = processor.download_pdf()
            
            # Verify
            assert result == temp_path
            assert os.path.exists(temp_path)
            mock_get.assert_called_once()
            mock_response.iter_content.assert_called_once()
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @patch('fitz.open')
    def test_extract_text(self, mock_open):
        """Test extracting text from a PDF."""
        # Setup mock
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test page content"
        mock_doc.__enter__.return_value = [mock_page]
        mock_open.return_value = mock_doc
        
        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Create an empty file
            with open(temp_path, 'wb') as f:
                f.write(b"dummy pdf content")
            
            # Test extract text
            processor = PDFProcessor(pdf_path=temp_path)
            result = processor.extract_text()
            
            # Verify
            assert "Page 1" in result
            assert "Test page content" in result
            mock_open.assert_called_once_with(temp_path)
            mock_page.get_text.assert_called_once()
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_chunk_text(self):
        """Test chunking text into smaller pieces."""
        # Sample text with page markers
        text = (
            "Page 1:\nThis is the first page content. It contains some information.\n\n"
            "This is a second paragraph on the first page.\n\n"
            "Page 2:\nThis is the second page content. It also has information.\n\n"
            "This is a second paragraph on the second page.\n\n"
        )
        
        processor = PDFProcessor()
        chunks = processor.chunk_text(text, chunk_size=100, overlap=20)
        
        # Verify
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all("page" in chunk["metadata"] for chunk in chunks)
        
        # Check first chunk
        assert chunks[0]["metadata"]["page"] == 1
        assert "first page content" in chunks[0]["text"]
