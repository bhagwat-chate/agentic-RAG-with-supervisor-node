from config.logging_config import *

from src.workflows.graph_builder import BuildGraph
from langchain_core.messages import SystemMessage
from src.constant.constant import *
import logging

logger = logging.getLogger(__name__)


def run_agentic_workflow():
    user_messages = [
        "Tell me how GenAI is used in healthcare.",
        "Tell me latest information about Dr. APJ Abdul Kalam from web world?",
        "Do you think India will be able to develop their own LLM? Agentic AI is fascinating!"
    ]

    for message in user_messages:
        logger.info(f"Processing query: {message}")

        state = {
            'messages': [SystemMessage(content=message)],
            'last_route': '',
            'retry_count': 0
        }

        try:
            app = BuildGraph(state).build_graph()
            output = app.stream(state)

            for event in output:
                if isinstance(event, tuple) and len(event) == 2:
                    node_name, node_state = event
                    logger.info(f"[{node_name}] → {node_state}")
                else:
                    logger.info(f"[STREAM EVENT] → {event}")

        except Exception as e:
            logger.exception(f"Error during workflow execution: {e}")


if __name__ == '__main__':
    logger.info(EXECUTION_START)
    run_agentic_workflow()
    logger.info(EXECUTION_END)
