from src.models.agent_state import AgentState
from langchain_core.messages import SystemMessage
from config.config_entity import ConfigEntity
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class ValidationNode:

    def __init__(self):
        self.config = ConfigEntity()

    def route_validation(self, state: AgentState) -> str:
        logger.info(EXECUTION_START)
        logger.info(EXECUTION_END)

        return "pass" if state.get("validation_passed") else "fail"

    def validation_router(self, state: AgentState) -> AgentState:
        try:
            logger.info(EXECUTION_START)

            # Get retry count
            retry_count = state.get("retry_count", 0)

            # Custom validation logic here
            is_valid = self.some_validation(state)  # your actual logic
            state["validation_passed"] = is_valid

            # Update retry count if failed
            if not is_valid:
                retry_count += 1
                state["retry_count"] = retry_count
                if retry_count >= self.config.max_llm_retries:  # 💥 Break loop after 3 retries
                    print("Max retries exceeded.")
                    state["validation_passed"] = True  # Treat as valid or trigger alternate exit
                    state["messages"].append(SystemMessage(content="Max retries hit. Forcing pass."))
            else:
                state["retry_count"] = 0  # reset on success

            logger.info(EXECUTION_END)

            return state

        except Exception as e:
            raise e

    def some_validation(self, state: AgentState) -> bool:
        try:
            logger.info(EXECUTION_START)

            last_msg = state["messages"][-1]
            if not isinstance(last_msg, SystemMessage):
                return False
            response = last_msg.content

            logger.info(EXECUTION_END)

            # Example: basic sanity checks
            return (
                    len(response.strip()) > 20 and
                    not response.strip().endswith("...") and
                    "error" not in response.lower()
            )

        except Exception as e:
            raise e
