import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Danh sách tin nhắn, dùng operator.add để append tin nhắn mới vào lịch sử
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Ai là người vừa thực hiện hành động cuối cùng (Supervisor hay Worker?)
    next: str
