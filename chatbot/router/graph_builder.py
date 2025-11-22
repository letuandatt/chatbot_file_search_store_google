from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from chatbot.core.state import AgentState
from chatbot.router.supervisor import create_supervisor_node
from chatbot.llm.agent_react import create_agent_executor


# Tạo một Worker Node đơn giản
def create_worker_node(agent_executor, name):
    def worker_node(state: AgentState):
        # Lấy tin nhắn cuối cùng
        result = agent_executor.invoke({"question": state["messages"][-1].content})
        return {
            "messages": [AIMessage(content=result["output"], name=name)]
        }

    return worker_node


def build_multi_agent_graph(text_llm, tools_policy, tools_personal):
    """
    Xây dựng đồ thị Multi-Agent: Supervisor -> [Worker] -> End
    """
    members = ["PolicyResearcher", "PersonalAnalyst"]

    # 1. Tạo Supervisor
    supervisor_chain = create_supervisor_node(text_llm, members)

    # 2. Tạo Workers (Sử dụng lại ReAct Executor cũ nhưng chia nhỏ tools)
    # Worker 1: Chỉ cầm tool Policy
    policy_agent = create_agent_executor(text_llm, tools_policy)
    policy_node = create_worker_node(policy_agent, "PolicyResearcher")

    # Worker 2: Chỉ cầm tool Personal + List Files
    personal_agent = create_agent_executor(text_llm, tools_personal)
    personal_node = create_worker_node(personal_agent, "PersonalAnalyst")

    # 3. Khởi tạo Graph
    workflow = StateGraph(AgentState)

    # Thêm các Nodes
    workflow.add_node("Supervisor", supervisor_chain)
    workflow.add_node("PolicyResearcher", policy_node)
    workflow.add_node("PersonalAnalyst", personal_node)

    # 4. Định nghĩa Edges (Luồng đi)
    # Từ Supervisor -> Quyết định đi đâu
    workflow.set_entry_point("Supervisor")

    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],  # Lấy giá trị 'next' từ output của Supervisor
        {
            "PolicyResearcher": "PolicyResearcher",
            "PersonalAnalyst": "PersonalAnalyst",
            "FINISH": END
        }
    )

    # Sau khi Worker làm xong, quay lại Supervisor (để kiểm tra hoặc kết thúc)
    # Hoặc trong mô hình đơn giản này, ta cho Worker về END luôn để trả lời User.
    workflow.add_edge("PolicyResearcher", END)
    workflow.add_edge("PersonalAnalyst", END)

    return workflow.compile()
