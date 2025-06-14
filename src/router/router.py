from src.workflows.graph_builder import AgentState
from langchain_core.messages import SystemMessage


class Router:

    def router_node(self, state: AgentState):
        try:
            messages = state['messages'][-1].content.lower()

            if 'llm' in messages:
                next_node = 'llm'
            elif 'web' in messages:
                next_node = 'web'
            elif 'rag' in messages:
                next_node = 'rag'
            else:
                raise ValueError(f"Invalid node request in route: {messages}")

            state = {
                     'messages': [SystemMessage(content=next_node)],
                     'validation_passed': False,
                     'last_route': next_node,
                     'retry_count': 0
                     }

            return state

        except Exception as e:
            raise e
