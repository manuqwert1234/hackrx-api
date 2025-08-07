import os
import json
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
API_KEY = os.getenv("API_KEY", "f8be7344e6c2cc6435ec3807f6750f7c5e8a8045d6d2a9e1b4ce6b3f3c09a534")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Test document URL (replace with a valid document URL for testing)
TEST_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

async def test_process_document():
    """Test the document processing and question answering endpoint."""
    url = f"{BASE_URL}/hackrx/run"
    
    # Test questions
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
    
    payload = {
        "documents": TEST_DOCUMENT_URL,
        "questions": questions
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Sending document processing request...")
            async with session.post(url, headers=HEADERS, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Error: {response.status} - {error}")
                    return None
                
                result = await response.json()
                task_id = result.get("task_id")
                logger.info(f"Task started with ID: {task_id}")
                return task_id
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

async def check_task_status(task_id: str):
    """Check the status of a processing task."""
    url = f"{BASE_URL}/status/{task_id}"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=HEADERS) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Error: {response.status} - {error}")
                    return None
                
                return await response.json()
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

async def wait_for_task_completion(task_id: str, max_attempts: int = 30, delay: int = 5):
    """Wait for a task to complete, checking status periodically."""
    for attempt in range(max_attempts):
        status = await check_task_status(task_id)
        
        if not status:
            logger.error("Failed to get task status")
            return None
            
        task_status = status.get("status")
        progress = status.get("progress", 0) * 100
        
        logger.info(f"Task status: {task_status} (Progress: {progress:.1f}%)")
        
        if task_status in ["completed", "failed"]:
            return status
            
        await asyncio.sleep(delay)
    
    logger.warning("Max attempts reached, task did not complete in time")
    return None

async def main():
    """Run the test workflow."""
    logger.info("Starting API test...")
    
    # Test 1: Process document and answer questions
    task_id = await test_process_document()
    if not task_id:
        logger.error("Failed to start document processing task")
        return
    
    # Test 2: Check task status and wait for completion
    logger.info("Waiting for task to complete...")
    result = await wait_for_task_completion(task_id)
    
    if not result:
        logger.error("Task did not complete successfully")
        return
    
    # Print the results
    logger.info("\n=== Test Results ===")
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Message: {result.get('message')}")
    
    if result.get("status") == "completed":
        answers = result.get("result", {}).get("answers", [])
        for i, answer in enumerate(answers, 1):
            logger.info(f"\nQuestion {i}: {answer.get('question')}")
            logger.info(f"Answer: {answer.get('answer')}")
            logger.info(f"Confidence: {answer.get('confidence', 0) * 100:.1f}%")
            
            # Print evidence if available
            evidence = answer.get("evidence", [])
            if evidence:
                logger.info("\nEvidence:")
                for j, ev in enumerate(evidence, 1):
                    logger.info(f"  {j}. {ev[:200]}...")
            
            # Print explanation if available
            explanation = answer.get("explanation")
            if explanation:
                logger.info(f"\nExplanation: {explanation}")
    else:
        logger.error(f"Task failed: {result.get('message')}")

if __name__ == "__main__":
    asyncio.run(main())
