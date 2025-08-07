from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model():
    """
    Downloads and caches the sentence-transformer model from Hugging Face.
    This is a one-time operation to prevent timeouts during server startup.
    """
    model_name = 'all-MiniLM-L6-v2'
    try:
        logger.info(f"Attempting to download and cache model: {model_name}")
        SentenceTransformer(model_name)
        logger.info(f"Successfully downloaded and cached model: {model_name}")
        logger.info("You can now start the main application server.")
    except Exception as e:
        logger.error(f"Failed to download model. Error: {e}")
        logger.error("Please check your internet connection and try running the script again.")

if __name__ == "__main__":
    download_model()
