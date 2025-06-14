import os

from dotenv import load_dotenv

load_dotenv()


class ConfigEntity:
    def __init__(self):

        self.google_api_key = os.getenv('GOOGLE_API_KEY', '')
        self.google_embedding_model = os.getenv('GOOGLE_EMBEDDING_MODEL', '')
        self.google_inference_LLM = os.getenv('GOOGLE_INFERENCE_LLM', '')
        self.llm_temperature = os.getenv('LLM_TEMPERATURE', '')
        self.llm_top_p = os.getenv('LLM_TOP_P', '')

        self.pc_index = os.getenv('PINECONE_INDEX_NAME', '')
        self.pc_namespace = os.getenv('PINECONE_NAMESPACE', '')
        self.pc_api_key = os.getenv('PINECONE_API_KEY', '')
        self.pc_index_cloud_region = os.getenv('PINECONE_INDEX_ENVIRONMENT', '')
        self.pc_cloud_vendor = os.getenv('PINECONE_CLOUD', '')
        self.pc_index_metric = os.getenv('PINECONE_INDEX_METRIC', '')
        self.doc_chunk_size = int(os.getenv('DOCUMENT_CHUNK_SIZE', ''))
        self.doc_overlap_size = int(os.getenv('DOCUMENT_OVERLAP_SIZE', ''))
        self.pc_retriever_top_k = os.getenv('DOCUMENT_RETRIEVER_TOP_K', '')










