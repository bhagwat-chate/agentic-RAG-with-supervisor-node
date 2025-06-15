from langchain_pinecone import PineconeVectorStore
from config.config_entity import ConfigEntity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class RetrieverBuilder:
    def __init__(self):
        self.config_entity = ConfigEntity()

    def build(self):
        try:
            logger.info(EXECUTION_START)

            embedding_model_obj = GoogleGenerativeAIEmbeddings(model=self.config_entity.google_embedding_model)
            retriever = PineconeVectorStore.from_existing_index(index_name=self.config_entity.pc_index,
                                                                embedding=embedding_model_obj,
                                                                namespace=self.config_entity.pc_namespace
                                                                )
            retriever = retriever.as_retriever(search_kwargs={'k': int(self.config_entity.pc_retriever_top_k)})

            logger.info(EXECUTION_END)

            return retriever

        except Exception as e:
            raise e
