import os
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from src.workflows.llm_node import LLMNode
from src.workflows.rag_node import RAGNode
from src.workflows.validator_node import ValidationNode
from src.workflows.web_node import WebNode
from src.workflows.supervisor_node import SupervisorNode
from src.models.agent_state import AgentState
from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class BuildGraph:

    # def __init__(self, state: AgentState):
    #     self.state = state

    def build_graph(self):
        try:
            logger.info(EXECUTION_START)

            supervisor_node_obj = SupervisorNode()
            llm_node_obj = LLMNode()
            rag_node_obj = RAGNode()
            web_node_obj = WebNode()
            validation_node_obj = ValidationNode()
            workflow = StateGraph(AgentState)

            workflow.add_node('supervisor', supervisor_node_obj.classify_request)

            workflow.add_node('llm', llm_node_obj.llm_response)
            workflow.add_node('rag', rag_node_obj.rag_response)
            workflow.add_node('web', web_node_obj.web_response)

            workflow.add_node('validation_router', validation_node_obj.validation_router)

            workflow.add_edge(START, 'supervisor')

            workflow.add_conditional_edges('supervisor',
                                           supervisor_node_obj.get_route,
                                           {'llm': 'llm', 'web': 'web', 'rag': 'rag'})

            workflow.add_edge('llm', 'validation_router')
            workflow.add_edge('rag', 'validation_router')
            workflow.add_edge('web', 'validation_router')

            workflow.add_conditional_edges('validation_router',
                                           validation_node_obj.route_validation,
                                           {'pass': END, 'fail': 'supervisor'})

            workflow.add_edge('validation_router', END)

            app = workflow.compile()

            logger.info(EXECUTION_END)

            return app

        except Exception as e:
            raise e
