from src.models.agent_state import AgentState
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_entity import ConfigEntity
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class LLMNode:

    def __init__(self):
        self.config = ConfigEntity()

    def llm_response(self, state: AgentState):
        try:
            logger.info(EXECUTION_START)

            question = state['messages'][0].content

            model = ChatGoogleGenerativeAI(model=self.config.google_inference_LLM)

            response = model.invoke(question)

            state = {
                     'messages': [SystemMessage(content=response.content)],
                     'validation_passed': state.get('validation_passed', ''),
                     'last_route': 'llm',
                     'retry_count': 0
                     }

            logger.info(EXECUTION_END)

            return state

        except Exception as e:
            raise e
