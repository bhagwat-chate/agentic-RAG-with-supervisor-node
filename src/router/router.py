from src.models.agent_state import AgentState
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class Router:

    def router_node(self, state: AgentState):
        try:
            logger.info(EXECUTION_START)

            message = state['last_route']

            if 'rag' in message:
                logger.info("route: 'rag'")
                logger.info(EXECUTION_END)

                return 'rag'

            elif 'web' in message:
                logger.info("route: 'web'")
                logger.info(EXECUTION_END)

                return 'web'

            elif 'llm' in message:
                logger.info("route: 'llm'")
                logger.info(EXECUTION_END)

                return 'llm'

            else:
                logger.error(f"ERROR: {message}")

                raise ValueError(f"Unexpected message in router: {message}")

        except Exception as e:
            logger.error(f"ERROR: {e}")
            raise e
