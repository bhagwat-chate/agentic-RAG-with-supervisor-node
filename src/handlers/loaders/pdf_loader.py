import os
from langchain_community.document_loaders import PyPDFLoader
from config.config_entity import ConfigEntity
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class PDFLoaderHandler:
    def __init__(self, file_path):
        self.file_path: str = file_path
        self.config_entity: ConfigEntity = ConfigEntity()

    def load_corpus(self) -> str:
        try:
            logger.info(EXECUTION_START)

            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"❌ File not found: {self.file_path}")

            loader = PyPDFLoader(self.file_path)
            pdf_documents = loader.load()[12:]

            pdf_corpus = ''
            for page in pdf_documents:
                pdf_corpus = pdf_corpus + page.page_content

            logger.info(EXECUTION_END)

            return pdf_corpus

        except Exception as e:
            logger.exception(f"{e}")
            raise e
