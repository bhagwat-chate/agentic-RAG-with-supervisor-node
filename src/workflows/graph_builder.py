import operator

from IPython.display import Image, display
from typing_extensions import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph
from langgraph.graph import START, END
from src.workflows.llm_node import LLMNode
from src.workflows.rag_node import RAGNode
from src.workflows.validator_node import ValidationNode
from src.workflows.web_node import WebNode
from src.workflows.supervisor_node import SupervisorNode
from src.router.router import Router
from src.models.agent_state import AgentState


class BuildGraph:

    def __init__(self, state: AgentState):
        self.state = state

    def build_graph(self):

        supervisor_node_obj = SupervisorNode()
        llm_node_obj = LLMNode()
        rag_node_obj = RAGNode()
        web_node_obj = WebNode()
        validation_node_obj = ValidationNode()
        router_obj = Router()
        workflow = StateGraph(AgentState)

        workflow.add_node('supervisor', supervisor_node_obj.classify_request)
        workflow.add_node('llm', llm_node_obj.llm_response)
        workflow.add_node('rag', rag_node_obj.rag_response)
        workflow.add_node('web', web_node_obj.web_response)
        workflow.add_node('validate', validation_node_obj.route_validation)

        workflow.add_edge(START, 'supervisor')

        workflow.add_conditional_edges('supervisor', validation_node_obj.validation_response, {'llm': 'llm', 'web': 'web', 'rag': 'rag'})

        workflow.add_edge('llm', 'validate')
        workflow.add_edge('rag', 'validate')
        workflow.add_edge('web', 'validate')

        workflow.add_conditional_edges('validate', router_obj.router_node, {'pass': END, 'fail': 'supervisor'})

        workflow.add_edge('validate', END)

        app = workflow.compile()

        display(Image(app.get_graph().draw_mermaid_png()))

        return app
