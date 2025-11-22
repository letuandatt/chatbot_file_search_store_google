from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser


def create_supervisor_node(llm, members: list[str]):
    system_prompt = (
        "Bạn là người quản lý giám sát (Supervisor) của một hệ thống chatbot CUSC.\n"
        "Nhiệm vụ của bạn là đọc cuộc hội thoại và quyết định xem ai sẽ hành động tiếp theo.\n"
        "Các thành viên (Workers) gồm: {members}.\n"
        " - Dùng 'PolicyResearcher' cho các câu hỏi về quy định chung, thủ tục, ISO.\n"
        " - Dùng 'PersonalAnalyst' cho các câu hỏi về file tài liệu người dùng tải lên.\n"
        " - Nếu đã có câu trả lời cuối cùng hoặc câu hỏi chào hỏi xã giao, hãy chọn 'FINISH'."
    )

    options = ["FINISH"] + members

    # Schema cho function calling
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            # "title": "routeSchema",  <-- Đã xóa dòng này để tránh Warning
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next Role",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    # --- SỬA LỖI TẠI ĐÂY ---
    # Đổi ("system", "Với tình huống...") thành ("human", "Với tình huống...")
    # Vì Gemini không cho phép SystemMessage nằm sau HumanMessage
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Dựa vào nội dung trên, ai nên hành động tiếp theo? Chọn một trong: {options}"),
    ]).partial(options=str(options), members=", ".join(members))

    # Supervisor chain
    supervisor_chain = (
            prompt
            | llm.bind_tools(tools=[function_def], tool_choice="route")
            | JsonOutputKeyToolsParser(key_name="route", first_tool_only=True)
    )

    return supervisor_chain
