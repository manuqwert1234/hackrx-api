import os
import logging
import tempfile
from typing import List, Dict, Any, Optional
import requests
from pathlib import Path
import magic
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd
from io import BytesIO
import io
import mimetypes

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document processing for various file types including PDF, DOCX, and emails.
    """
    
    def __init__(self):
        self.supported_mime_types = {
            'application/pdf': self._process_pdf,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._process_docx,
            'application/msword': self._process_doc,
            'text/plain': self._process_text,
            'text/html': self._process_html,
        }
    
    async def process_document(self, document_url: str) -> str:
        """
        Download and process a document from a URL.
        
        Args:
            document_url: URL to the document
            
        Returns:
            str: Extracted text content from the document
        """
        try:
            # Download the document
            response = requests.get(document_url, stream=True)
            response.raise_for_status()
            
            # Get content type
            content_type = response.headers.get('content-type', '').split(';')[0].strip()
            
            # If content type is not specific, try to detect from content
            if not content_type or content_type == 'application/octet-stream':
                content = response.content
                mime = magic.Magic(mime=True)
                content_type = mime.from_buffer(content[:1024])
            
            # Process based on content type
            process_func = self.supported_mime_types.get(content_type)
            if process_func:
                return await process_func(response.content)
            else:
                logger.warning(f"Unsupported content type: {content_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise Exception(f"Failed to process document: {str(e)}")
    
    async def _process_pdf(self, content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    async def _process_docx(self, content: bytes) -> str:
        """Extract text from DOCX content."""
        try:
            doc_file = io.BytesIO(content)
            doc = Document(doc_file)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error processing DOCX: {str(e)}")
            raise
    
    async def _process_doc(self, content: bytes) -> str:
        """Extract text from DOC content (legacy Word format)."""
        try:
            # For .doc files, we'll use textract which supports more formats
            import textract
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                text = textract.process(temp_file_path).decode('utf-8')
                return text
            finally:
                os.unlink(temp_file_path)
        except Exception as e:
            logger.error(f"Error processing DOC: {str(e)}")
            raise
    
    async def _process_text(self, content: bytes) -> str:
        """Process plain text content."""
        try:
            return content.decode('utf-8')
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise
    
    async def _process_html(self, content: bytes) -> str:
        """Extract text from HTML content."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            return soup.get_text(separator="\n", strip=True)
        except Exception as e:
            logger.error(f"Error processing HTML: {str(e)}")
            # Fallback to raw text if HTML parsing fails
            return await self._process_text(content)
