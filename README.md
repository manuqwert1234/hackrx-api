# LLM-Powered Intelligent Query–Retrieval System

A FastAPI-based system for processing documents and answering questions using LLM (Gemini) with semantic search capabilities.

## Features

- Process various document types (PDF, DOCX, TXT, HTML)
- Semantic search using FAISS for efficient similarity search
- Asynchronous processing of documents and questions
- RESTful API with JWT authentication
- Detailed response with confidence scores and evidence
- Progress tracking for long-running tasks

## Prerequisites

- Python 3.9+
- pip (Python package manager)
- Gemini API key (from Google AI Studio)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```env
   # Required
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional (default shown)
   API_KEY=f8be7344e6c2cc6435ec3807f6750f7c5e8a8045d6d2a9e1b4ce6b3f3c09a534
   ```

## Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## API Endpoints

### Process Document and Answer Questions

```http
POST /api/v1/hackrx/run
Authorization: Bearer <api_key>
Content-Type: application/json

{
  "documents": "https://example.com/document.pdf",
  "questions": ["What is the main topic?", "What are the key points?"]
}
```

### Check Task Status

```http
GET /api/v1/status/{task_id}
Authorization: Bearer <api_key>
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── document_processor.py  # Handles document processing
│   ├── embeddings.py          # FAISS-based semantic search
│   ├── llm_service.py         # Gemini LLM integration
│   └── query_processor.py     # Query processing logic
├── .env.example              # Example environment variables
├── main.py                   # FastAPI application
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Your Gemini API key |
| `API_KEY` | No | API key for authentication (default provided) |

## Error Handling

The API returns appropriate HTTP status codes and JSON error responses:

- `400 Bad Request`: Invalid input data
- `401 Unauthorized`: Missing or invalid API key
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Testing

To test the API, you can use the provided test script:

```bash
python test_api.py
```

## Deployment

For production deployment, consider using:

- Gunicorn with Uvicorn workers
- Environment variable management
- Proper logging and monitoring
- CORS configuration for production

## License

This project is licensed under the MIT License - see the LICENSE file for details.
