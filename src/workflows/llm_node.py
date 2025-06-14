from src.workflows.graph_builder import AgentState
from langchain_core.messages import SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_entity import ConfigEntity


class LLMNode:

    def __init__(self):
        self.config = ConfigEntity()

    def llm_node(self, state: AgentState):
        try:
            question = state['messages'][0].content

            model = ChatGoogleGenerativeAI(model=self.config.google_inference_LLM)

            response = model.invoke(question)

            state = {
                     'messages': [SystemMessage(content=response.content)],
                     'validation_passed': False,
                     'last_route': 'llm',
                     'retry_count': 0
                     }

            return state

        except Exception as e:
            raise e
