import logging
import json
from typing import List, Dict, Any, Optional
import os
import re
from pathlib import Path
from datetime import datetime

from .document_processor import DocumentProcessor
from .embeddings import DocumentEmbedder, SearchResult
from .llm_service import LLMService

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Handles the end-to-end process of querying documents and generating responses.
    """
    
    def __init__(self, llm_service: LLMService, cache_dir: str = ".cache"):
        """
        Initialize the query processor with required services.
        
        Args:
            llm_service: Initialized LLM service for processing queries
            cache_dir: Directory to store cached embeddings and processed documents
        """
        self.llm_service = llm_service
        self.document_processor = DocumentProcessor()
        self.embedder = DocumentEmbedder()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache for loaded document indices
        self.document_indices = {}
    
    async def process_query(self, query: str, document_url: str) -> Dict[str, Any]:
        """
        Process a query against a document and return a structured response.
        
        Args:
            query: The user's natural language query
            document_url: URL to the document to query against
            
        Returns:
            Dict containing the answer and metadata
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Check if we have a cached index for this document
            doc_hash = self._get_document_hash(document_url)
            index_path = self.cache_dir / doc_hash
            
            if index_path.with_suffix('.index').exists():
                # Load existing index
                logger.info(f"Loading cached index for document: {doc_hash}")
                self.embedder = DocumentEmbedder.load_index(str(index_path))
            else:
                # Process document and create index
                logger.info(f"Processing document: {document_url}")
                text = await self.document_processor.process_document(document_url)
                
                if not text.strip():
                    raise ValueError("Document processing returned empty content")
                
                # Chunk the document
                chunks = self._chunk_document(text)
                
                # Build the index
                self.embedder.build_index(chunks)
                
                # Cache the index for future use
                self.embedder.save_index(str(index_path))
            
            # Step 2: Extract query intent
            intent = await self.llm_service.extract_query_intent(query)
            logger.info(f"Extracted query intent: {intent}")
            
            # Step 3: Semantic search for relevant context
            search_results = self.embedder.search(query, k=5)
            
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "confidence": 0.0,
                    "evidence": [],
                    "explanation": "No relevant context found in the document.",
                    "sources": []
                }
            
            # Extract context from search results
            context = [result.content for result in search_results]
            sources = [{"content": result.content, "score": result.score} 
                      for result in search_results]
            
            # Step 4: Generate answer using LLM
            llm_response = await self.llm_service.generate_answer(query, context)
            
            # Step 5: Generate explanation if not provided
            if "explanation" not in llm_response or not llm_response["explanation"]:
                llm_response["explanation"] = await self.llm_service.generate_explanation(
                    query, llm_response["answer"], context
                )
            
            # Add sources to the response
            llm_response["sources"] = sources
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "answer": "I encountered an error while processing your request.",
                "confidence": 0.0,
                "evidence": [],
                "explanation": str(e),
                "sources": []
            }
    
    def _chunk_document(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split a document into overlapping chunks of text.
        
        Args:
            text: The full text to chunk
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If adding this paragraph would exceed the chunk size, finalize current chunk
            if current_length + para_length > max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                
                # Start new chunk with overlap from previous chunk
                if overlap > 0 and chunks:
                    # Get last chunk and take the last 'overlap' characters
                    last_chunk = chunks[-1]
                    overlap_text = last_chunk[-overlap:]
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _get_document_hash(self, document_url: str) -> str:
        """Generate a unique hash for a document URL."""
        import hashlib
        return hashlib.md5(document_url.encode('utf-8')).hexdigest()


# Example usage
async def example_usage():
    """Example of how to use the QueryProcessor."""
    # Initialize services
    llm_service = LLMService(api_key="your_gemini_api_key")
    processor = QueryProcessor(llm_service)
    
    # Example document URL (replace with actual URL)
    document_url = "https://example.com/sample.pdf"
    
    # Example query
    query = "What are the key terms of this agreement?"
    
    # Process the query
    response = await processor.process_query(query, document_url)
    
    # Print the result
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
