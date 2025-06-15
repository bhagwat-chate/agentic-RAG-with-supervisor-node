from langchain.prompts import PromptTemplate
from config.config_entity import ConfigEntity
from pydantic import BaseModel
from src.models.agent_state import AgentState
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class TopicSelectionParser(BaseModel):
    answer: str


class PromptBuilder:

    def __init__(self, retriever):
        self.retriever = retriever
        self.config = ConfigEntity()

    def build(self, state: AgentState):
        try:
            logger.info(EXECUTION_START)

            prompt = PromptTemplate(input_variables=['context', 'question'], template="""
                                You are an assistant for question-answering tasks. Use the following context to answer the question.
                                Always respond **ONLY** in the following JSON format:
                                {{
                                "answer: "Your concise answer here"
                                }}
                                
                                context:
                                {context}
                                
                                question:
                                {question}
                            """)

            logger.info(EXECUTION_END)

            return prompt

        except Exception as e:
            logger.error(f"ERROR: {e}")
            raise e
