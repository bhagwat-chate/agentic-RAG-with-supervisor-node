from src.models.agent_state import AgentState
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from pydantic import Field
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_entity import ConfigEntity
from langchain_core.messages import SystemMessage


class TopicSelectionParser(BaseModel):
    Topic: str = Field(description='Topic selected')
    Reasoning: str = Field(description='The reasoning behind the topic selection')


class SupervisorNode:

    def __init__(self):
        self.config = ConfigEntity()

    def classify_request(self, state: AgentState):
        question = state['messages'][0]

        template = """
                    You are a classification agent. Classify the following user question into one of the three following category ONLY: [rag, web, llm].
                    
                    **Definitions:**
                    - rag: The question can be better answered using retrival from external knowledge or documents.
                    - web: The question required fresh or current information from the internet.
                    - llm: The question can be answered by the language model's own reasoning or general knowledge.
                    
                    You must answer with exactly one word: either rag, llm, or web.
                    
                    User question: {question}
                    {format_instructions}
                    """
        parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

        prompt = PromptTemplate(template=template,
                                input_variables=['question'],
                                partial_variables={'format_instructions': parser.get_format_instructions()}
                                )
        model = ChatGoogleGenerativeAI(model=self.config.google_inference_LLM)

        chain = prompt | model | parser

        response = chain.invoke({'question': question})

        state = {
            'messages': [SystemMessage(content=response.Topic)],
            'validation_passed': False,
            'last_route': state.get('last_route', ''),
            'retry_count': state.get('retry_count', 0)
        }

        return state
