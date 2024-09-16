from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence, List

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

class AgentState_er(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int = 0
    max_revisions: int = 2
