import os
from typing import List
from config.config_entity import ConfigEntity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.handlers.loaders.pdf_loader import PDFLoaderHandler
from src.handlers.chunking.text_splitter import TextSplitter
from uuid import uuid4


class RAGService:
    def __init__(self):
        self.config_entity = ConfigEntity()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.file_path = os.path.join(self.project_root, "data", "raw_data", "agentic-ai-system.pdf")

    def run_rag_pipeline(self):
        try:
            pdf_loader_obj = PDFLoaderHandler(file_path=self.file_path)
            text = pdf_loader_obj.load_corpus()

            text_splitter_obj = TextSplitter(corpus_str=text)
            chunks_lst = text_splitter_obj.split()

        except Exception as e:
            raise e

    def embed_documents(self, chunks: List[str]) -> None:
        try:
            embeddings_google = GoogleGenerativeAIEmbeddings(model=self.config_entity.google_embedding_model)
            doc_embeddings = embeddings_google.embed_documents(chunks)
            embedding_dimension = len(doc_embeddings[0])
            print(f'✅ documents embeddings created, len of embedding: {len(doc_embeddings[11])}')

            pc = Pinecone(api_key=self.config_entity.pc_api_key, environment=self.config_entity.pc_index_cloud_region)

            if not pc.has_index(self.config_entity.pc_index):
                pc.create_index(name=self.config_entity.pc_index,
                                dimension=embedding_dimension,
                                metric=self.config_entity.pc_index_metric.lower(),
                                spec=ServerlessSpec(cloud=self.config_entity.pc_cloud_vendor, region=self.config_entity.pc_index_cloud_region)
                                )
            pc_index = pc.Index(self.config_entity.pc_index)

            pc_vector_space = PineconeVectorStore(index=pc_index, embedding=embeddings_google)

            uuids = [str(uuid4()) for _ in range(len(doc_embeddings))]

            batch_size = 500
            start = 0
            end = batch_size

            chunks_doc = [Document(page_content=page, metadata={'page': idx+1}) for idx, page in enumerate(chunks)]
            while start < len(chunks_doc):
                if end > len(chunks_doc):
                    end = len(chunks_doc)

                pc_vector_space.add_documents(documents=chunks_doc[start:end], ids=uuids[start:end])

                start = end
                end = end + batch_size

            print(f'✅ loaded docs {len(chunks_doc)}')

        except Exception as e:
            raise e


if __name__ == '__main__':
    rag_service_obj = RAGService()
    pdf_corpus = rag_service_obj.run_rag_pipeline()
    # chunks = rag_service_obj.clean_and_split(pdf_corpus)
    # rag_service_obj.embed_documents(chunks)

    print('done')










