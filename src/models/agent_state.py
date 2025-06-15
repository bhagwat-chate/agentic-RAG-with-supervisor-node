import operator

from typing_extensions import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

from src.constant.constant import *
import logging
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    try:
        logger.info(EXECUTION_START)

        messages: Annotated[Sequence[BaseMessage], operator.add]
        validation_passed: bool = False
        last_route: str = ''
        retry_count: int = 0

        logger.info(EXECUTION_END)

    except Exception as e:
        logger.exception(f"{e}")
        raise e
