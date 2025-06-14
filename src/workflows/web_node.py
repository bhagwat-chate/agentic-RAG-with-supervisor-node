from src.models.agent_state import AgentState
from langchain_core.messages import SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun


class WebNode:

    def web_response(self, state: AgentState):
        try:
            question = state['messages'][0].content

            search = DuckDuckGoSearchRun()

            response = search.invoke(question)

            state = {
                     'messages': [SystemMessage(content=response)],
                     'validation_passed': False,
                     'last_route': 'web',
                     'retry_count': 0
                     }
            return state

        except Exception as e:
            raise e
