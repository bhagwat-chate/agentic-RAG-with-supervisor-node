from config.logging_config import *  # <-- Must be first import

from src.services.rag_pipeline import RAGService

from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


def main():
    """
    Entry point for running the complete RAG pipeline:
    - Loads the document
    - Splits into chunks
    - Embeds the chunks
    - Uploads to vector store
    """

    try:
        logger.info(EXECUTION_START)

        rag_service = RAGService()
        rag_service.run_rag_pipeline()

        logger.info(EXECUTION_END)

    except Exception as e:
        logger.exception(f" ERROR: {e}")
        raise e


if __name__ == '__main__':
    main()
