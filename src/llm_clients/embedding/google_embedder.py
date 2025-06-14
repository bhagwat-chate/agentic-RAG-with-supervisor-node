from typing import List
from config.config_entity import ConfigEntity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


class GoogleEmbedder:
    def __init__(self):
        self.config_entity = ConfigEntity()

    def embed(self, chunks: List[str]):
        try:
            google_embedding = GoogleGenerativeAIEmbeddings(model=self.config_entity.google_embedding_model)
            embedding_lst = google_embedding.embed_documents(chunks)
            document_lst = [Document(page_content=page, metadata={'page': idx + 1})
                            for idx, page in enumerate(chunks)]

            return document_lst, embedding_lst, google_embedding

        except Exception as e:
            raise e
