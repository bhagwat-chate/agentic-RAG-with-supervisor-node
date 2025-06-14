from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config_entity import ConfigEntity
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser
from pydantic import Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage
from src.workflows.graph_builder import AgentState


class TopicSelectionParser(BaseModel):
    Topic: str = Field(description='selected topic')
    Reasoning: str = Field(description='Reasoning behind the topic selection')


class PromptBuilder:

    def __init__(self, retriever):
        self.retriever = retriever
        self.config = ConfigEntity()

    def format_doc(self, docs):
        return '\n\n'.join(doc for doc in docs)

    def build(self, state: AgentState):
        try:
            prompt = PromptTemplate(input_variables=['context', 'question'], template="""
                                You are an assistant for question-answering tasks. Use the following context to answer the question.
                                Always respond **ONLY** in the following JSON format:
                                {{
                                "answer: "Your concise answer here"
                                }}
                                
                                context:
                                {context}
                                
                                question:
                                {question}
                            """)
            question = state['messages'][0].content
            model = ChatGoogleGenerativeAI(self.config.google_inference_LLM)
            parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

            rag_chain = ({'context': self.retriever | self.format_doc, 'question': RunnablePassthrough()}
                         | prompt
                         | model
                         | PydanticOutputParser(pydantic_object=TopicSelectionParser))
            response = rag_chain.invoke(question)

            state = {'messages': [SystemMessage(content=response.answer)],
                     'validation_passed': state.get('validation_passed', ''),
                     'last_route': state.get('last_route', ''),
                     'retry_count': state.get('retry_count', 0)
                     }
            return state

        except Exception as e:
            raise e
