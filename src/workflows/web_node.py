from src.models.agent_state import AgentState
from langchain_core.messages import SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class WebNode:

    def web_response(self, state: AgentState):
        try:
            logger.info(EXECUTION_START)

            question = state['messages'][0].content

            search = DuckDuckGoSearchRun()

            response = search.invoke(question)

            state = {
                     'messages': [SystemMessage(content=response)],
                     'validation_passed': False,
                     'last_route': 'web',
                     'retry_count': 0
                     }

            logger.info(EXECUTION_END)

            return state

        except Exception as e:
            raise e
