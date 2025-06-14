import os
from langchain.document_loaders import PyPDFLoader
from config.config_entity import ConfigEntity


class PDFLoaderHandler:
    def __init__(self, file_path):
        self.file_path: str = file_path
        self.config_entity: ConfigEntity = ConfigEntity()

    def load_corpus(self) -> str:
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"❌ File not found: {self.file_path}")

            loader = PyPDFLoader(self.file_path)
            pdf_documents = loader.load()[12:20]

            pdf_corpus = ''
            for page in pdf_documents:
                pdf_corpus = pdf_corpus + page.page_content

            return pdf_corpus
        except Exception as e:
            raise e
