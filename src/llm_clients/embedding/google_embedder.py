from typing import List
from config.config_entity import ConfigEntity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class GoogleEmbedder:
    def __init__(self):
        self.config_entity = ConfigEntity()

    def embed(self, chunks: List[str]):
        try:
            logger.info(EXECUTION_START)

            google_embedding = GoogleGenerativeAIEmbeddings(model=self.config_entity.google_embedding_model)
            embedding_lst = google_embedding.embed_documents(chunks)
            document_lst = [Document(page_content=page, metadata={'page': idx + 1})
                            for idx, page in enumerate(chunks)]

            logger.info(EXECUTION_END)

            return document_lst, embedding_lst, google_embedding

        except Exception as e:
            logger.exception(f"{e}")
            raise e
