from config.config_entity import ConfigEntity
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(self, corpus_str):
        self.corpus_str = corpus_str
        self.config_entity = ConfigEntity()

    def clean(self):
        pass

    def split(self):
        try:

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.config_entity.doc_chunk_size,
                                                           chunk_overlap=self.config_entity.doc_overlap_size,
                                                           separators=['\n\n', '\n', '.', ' '])
            chunks = text_splitter.split_text(self.corpus_str)

            print(f'✅ corpus split into {len(chunks)} chunks')

            return chunks

        except Exception as e:
            raise e