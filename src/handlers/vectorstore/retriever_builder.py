from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from config.config_entity import ConfigEntity
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class PineconeRetriever:
    def __init__(self):
        self.config_entity = ConfigEntity()

    def retriever_builder(self):
        try:
            google_embedding_obj = GoogleGenerativeAIEmbeddings(model=self.config_entity.google_embedding_model)
            pc_vector_space = PineconeVectorStore(index=self.config_entity.pc_index, embedding=google_embedding_obj)

            retriever = pc_vector_space.as_retriever(kwargs={'k': int(self.config_entity.pc_retriever_top_k)})

            return retriever
        except Exception as e:
            raise e
