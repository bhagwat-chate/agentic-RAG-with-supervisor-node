import os

from config.config_entity import ConfigEntity

from src.handlers.vectorstore.pinecone_loader import PineconeLoader
from src.handlers.vectorstore.retriever_builder import RetrieverBuilder
from src.handlers.loaders.pdf_loader import PDFLoaderHandler
from src.handlers.chunking.text_splitter import TextSplitter
from src.llm_clients.embedding.google_embedder import GoogleEmbedder
from src.prompt_engineering.prompt_builder import PromptBuilder

import warnings
warnings.filterwarnings('ignore')


class RAGService:
    def __init__(self):
        self.config_entity = ConfigEntity()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.file_path = os.path.join(self.project_root, "data", "raw_data", "agentic-ai-system.pdf")

    def run_rag_pipeline(self):
        try:
            pdf_loader_obj = PDFLoaderHandler(file_path=self.file_path)
            text = pdf_loader_obj.load_corpus()
            print('✅ complete: data load')

            text_splitter_obj = TextSplitter(corpus_str=text)
            chunks_lst = text_splitter_obj.split()
            print('✅ complete: text clean and split')

            google_embeder_obj = GoogleEmbedder()
            document_lst, embedding_lst, google_embedding = google_embeder_obj.embed(chunks=chunks_lst)
            print('✅ complete: google embedding')

            pinecone_loader_obj = PineconeLoader()
            pinecone_loader_obj.store(chunks=chunks_lst, document_lst=document_lst, embedding_lst=embedding_lst, embedding_obj=google_embedding)
            print('✅ complete: embedding load into vector store (pinecone)')

            retriever_builder_obj = RetrieverBuilder()
            retriever = retriever_builder_obj.build()
            print('✅ complete: retriever created')

            prompt_builder_obj = PromptBuilder(retriever)
            prompt_builder_obj.build()
            print('✅ complete: prompt creation')

        except Exception as e:
            raise e
