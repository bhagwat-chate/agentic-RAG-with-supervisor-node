from src.models.agent_state import AgentState
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_entity import ConfigEntity
from src.handlers.vectorstore.retriever_builder import RetrieverBuilder
from src.prompt_engineering.prompt_builder import PromptBuilder
from src.prompt_engineering.prompt_builder import TopicSelectionParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class RAGNode:

    def __init__(self):
        self.config = ConfigEntity()

    def format_doc(self, docs):
        logger.info(EXECUTION_START)
        logger.info(EXECUTION_END)

        return '\n\n'.join(doc.page_content for doc in docs)

    def rag_response(self, state: AgentState):
        try:
            logger.info(EXECUTION_START)

            question = state['messages'][0].content
            retriever = RetrieverBuilder().build()
            prompt_builder_obj = PromptBuilder(retriever)

            prompt = prompt_builder_obj.build()

            model = ChatGoogleGenerativeAI(model=self.config.google_inference_LLM)

            parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

            rag_chain = ({'context': retriever | self.format_doc, 'question': RunnablePassthrough()}
                         | prompt
                         | model
                         | parser)

            response = rag_chain.invoke(question)

            state = {
                     'messages': [SystemMessage(content=response.answer)],
                     'validation_passed': False,
                     'last_route': 'rag',
                     'retry_count': 0
                     }

            logger.info(EXECUTION_END)

            return state

        except Exception as e:
            raise e
