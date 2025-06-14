import logging
import os
import sys

# Setup basic logging (optional: move to logger.py for better control)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# Adjust the path for absolute imports when running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.services.rag_pipeline import RAGService


def main():
    """
    Entry point for running the complete RAG pipeline:
    - Loads the document
    - Splits into chunks
    - Embeds the chunks
    - Uploads to vector store
    """
    logging.info("🚀 RAG pipeline execution started")

    try:
        rag_service = RAGService()
        rag_service.run_rag_pipeline()
        logging.info("✅ RAG pipeline executed successfully")

    except Exception as e:
        logging.exception(f"❌ Pipeline execution failed: {str(e)}")


if __name__ == "__main__":
    main()
