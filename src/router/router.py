from src.models.agent_state import AgentState
from langchain_core.messages import SystemMessage


class Router:

    def router_node(self, state: AgentState):
        message = state['last_route']

        if 'rag' in message:
            return 'rag'
        elif 'web' in message:
            return 'web'
        elif 'llm' in message:
            return 'llm'
        else:
            raise ValueError(f"Unexpected message in router: {message}")
