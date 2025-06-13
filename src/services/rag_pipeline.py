import os
from typing import List
from config.config_entity import ConfigEntity
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class RAGService:
    def __init__(self):
        self.config_entity = ConfigEntity()
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.file_path = os.path.join(self.project_root, "data", "raw_data", "agentic-ai-system.pdf")

    def load_corpus(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ File not found: {self.file_path}")

        loader = PyPDFLoader(self.file_path)
        pdf_documents = loader.load()[12:]
        print(f"✅ Loaded {len(pdf_documents)} pages from {self.file_path}")

        pdf_corpus = ''
        for page in pdf_documents:
            pdf_corpus = pdf_corpus + page.page_content

        return pdf_corpus

    def clean_and_split(self, corpus: str) -> List[str]:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config_entity.doc_chunk_size, chunk_overlap=self.config_entity.doc_overlap_size, separators=['\n\n', '\n', '.', ' '])
            chunks = text_splitter.split_text(corpus)
            print(f'✅ corpus split into {len(chunks)} chunks')

            documents = [Document(page_content=page, metadata={'page': idx+1}) for idx, page in enumerate(chunks)]
            print(f'✅ corpus split into {len(documents)} documents')

            return documents

        except Exception as e:
            raise e


if __name__ == '__main__':
    rag_service_obj = RAGService()
    pdf_corpus = rag_service_obj.load_corpus()
    documents = rag_service_obj.clean_and_split(pdf_corpus)

    print('done')










