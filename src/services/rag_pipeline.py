import os

from config.config_entity import ConfigEntity

from src.handlers.vectorstore.pinecone_loader import PineconeLoader
from src.handlers.vectorstore.retriever_builder import RetrieverBuilder
from src.handlers.loaders.pdf_loader import PDFLoaderHandler
from src.handlers.chunking.text_splitter import TextSplitter
from src.llm_clients.embedding.google_embedder import GoogleEmbedder
from src.prompt_engineering.prompt_builder import PromptBuilder
from src.constant.constant import *
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.config_entity = ConfigEntity()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.file_path = os.path.join(self.project_root, "data", "raw_data", "agentic-ai-system.pdf")

    def run_rag_pipeline(self):
        try:
            logger.info(EXECUTION_START)

            pdf_loader_obj = PDFLoaderHandler(file_path=self.file_path)
            text = pdf_loader_obj.load_corpus()
            logger.info('✅ complete: data load')

            text_splitter_obj = TextSplitter(corpus_str=text)
            chunks_lst = text_splitter_obj.split()
            logger.info('✅ complete: text clean and split')

            google_embeder_obj = GoogleEmbedder()
            document_lst, embedding_lst, google_embedding = google_embeder_obj.embed(chunks=chunks_lst)
            logger.info('✅ complete: google embedding')

            pinecone_loader_obj = PineconeLoader()
            pinecone_loader_obj.store(chunks=chunks_lst, document_lst=document_lst, embedding_lst=embedding_lst, embedding_obj=google_embedding)
            logger.info('✅ complete: embedding loaded into vector store (pinecone)')

            retriever_builder_obj = RetrieverBuilder()
            retriever = retriever_builder_obj.build()
            logger.info('✅ complete: retriever created')

            prompt_builder_obj = PromptBuilder(retriever)
            prompt_builder_obj.build()
            logger.info('✅ complete: prompt creation')

            logger.info(EXECUTION_END)

        except Exception as e:
            logger.error(f"ERROR: {e}")
            raise e
