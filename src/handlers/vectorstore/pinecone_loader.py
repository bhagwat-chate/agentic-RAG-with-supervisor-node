from config.config_entity import ConfigEntity
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class PineconeLoader:
    def __init__(self):
        self.config_entity = ConfigEntity()

    def store(self, chunks, document_lst, embedding_lst, embedding_obj):
        try:
            logger.info(EXECUTION_START)

            embedding_dimension = len(embedding_lst[0])

            pc = Pinecone(api_key=self.config_entity.pc_api_key, environment=self.config_entity.pc_index_cloud_region)

            if not pc.has_index(self.config_entity.pc_index):
                pc.create_index(name=self.config_entity.pc_index,
                                metric=self.config_entity.pc_index_metric,
                                dimension=embedding_dimension,
                                spec=ServerlessSpec(cloud=self.config_entity.pc_cloud_vendor,
                                                    region=self.config_entity.pc_index_cloud_region)
                                )

            uuid_lst = [str(uuid4()) for _ in range(len(document_lst))]

            pc_index = pc.Index(self.config_entity.pc_index)
            pc_vector_space = PineconeVectorStore(embedding=embedding_obj, index=pc_index)

            batch_size = 500
            start = 0
            end = batch_size

            while start < len(chunks):
                if end > len(chunks):
                    end = len(chunks)

                pc_vector_space.add_documents(documents=document_lst[start:end],
                                              ids=uuid_lst[start:end],
                                              namespace=self.config_entity.pc_namespace)

                start = end
                end = end + batch_size

            logger.info(EXECUTION_END)

        except Exception as e:
            logger.exception(f"{e}")
            raise e
