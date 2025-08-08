import os
import logging
import hashlib
from typing import List

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Import custom modules
from app.llm_service import LLMService
from app.document_processor import DocumentProcessor
from app.embeddings import DocumentEmbedder, SearchResult

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
API_KEY = os.getenv("API_KEY", "f8be7344e6c2cc6435ec3807f6750f7c5e8a8045d6d2a9e1b4ce6b3f3c09a534")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_CACHE_DIR = "embedding_cache"

# Security
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System",
    description="API for processing documents and answering questions using LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DocumentRequest(BaseModel):
    documents: List[str] = Field(..., description="List of document URLs (only the first will be processed)")
    questions: List[str] = Field(..., description="List of questions to answer")

class AnswerResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers to the questions")

# Global services
llm_service: LLMService
document_processor: DocumentProcessor

@app.on_event("startup")
def startup_event():
    """Initialize services and create cache directory on startup."""
    global llm_service, document_processor
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    llm_service = LLMService(api_key=GEMINI_API_KEY)
    document_processor = DocumentProcessor()
    
    # Create cache directory if it doesn't exist
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    logger.info(f"Services initialized. Cache directory is '{EMBEDDING_CACHE_DIR}'.")

# Security dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# API Endpoints
@app.get("/")
async def root():
    return {"message": "LLM-Powered Intelligent Query–Retrieval System is running!"}

@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
async def run_queries(request: DocumentRequest, token: str = Depends(verify_token)):
    """
    Process a document, find relevant context, and answer questions synchronously.
    Caches document embeddings to avoid reprocessing.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No document URL provided.")
    doc_url = request.documents[0]
    doc_id = hashlib.md5(doc_url.encode()).hexdigest()
    index_path = os.path.join(EMBEDDING_CACHE_DIR, doc_id)

    try:
        if os.path.exists(f"{index_path}.ann"):
            logger.info(f"Loading cached index for document: {doc_url}")
            embedder = DocumentEmbedder.load_index(index_path)
        else:
            logger.info(f"No cache found. Processing document from URL: {doc_url}")
            # 1. Process document to get text
            document_text = await document_processor.process_document(doc_url)
            if not document_text or not document_text.strip():
                raise HTTPException(status_code=400, detail="Document is empty or could not be processed.")

            # 2. Chunk text and build index
            chunks = chunk_document(document_text)
            if not chunks:
                raise HTTPException(status_code=400, detail="Could not extract text chunks.")
            
            embedder = DocumentEmbedder()
            embedder.build_index(chunks)
            
            # 3. Save the new index to cache
            embedder.save_index(index_path)
            logger.info(f"Saved new index to cache: {index_path}")

        # 4. Process questions
        logger.info(f"Querying with {len(embedder.documents)} document chunks.")
        answers = []
        for question in request.questions:
            logger.info(f"Processing question: '{question}'")
            search_results: List[SearchResult] = embedder.search(question, k=3)
            context = [result.content for result in search_results]
            logger.info(f"Context for question '{question}': {context}")
            
            llm_response = await llm_service.generate_answer(question, context)
            answer = llm_response.get("answer", "Could not find an answer.")
            answers.append(answer)
            logger.info(f"Generated answer: '{answer}'")

        return AnswerResponse(answers=answers)

    except Exception as e:
        logger.error(f"Error during query processing for {doc_url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def chunk_document(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Splits a document into overlapping chunks of text."""
    if not text:
        return []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_length = len(para)
        if current_length + para_length > max_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(para)
        current_length += para_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
