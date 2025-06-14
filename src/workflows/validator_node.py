from src.models.agent_state import AgentState
from langchain_core.messages import SystemMessage
from config.config_entity import ConfigEntity


class ValidationNode:

    def __init__(self):
        self.config = ConfigEntity()

    def validation_response(self, state: AgentState):
        try:
            retry_count = state.get("retry_count", 0)

            is_valid = True

            state = {
                'messages': [SystemMessage(content='validation passed')],
                'validation_passed': is_valid,
                'last_route': state.get('last_route', ''),
                'retry_count': state.get('retry_count', retry_count)
            }

            return state

        except Exception as e:
            raise e

    def route_validation(self, state: AgentState):
        return 'pass' if state.get('validation_passed', '') == 'pass' else 'fail'
