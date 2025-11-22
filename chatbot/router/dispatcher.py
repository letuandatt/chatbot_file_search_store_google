from chatbot.services.rag_pipeline import AdvancedRagPipeline
from chatbot.tools.tool_search_policy import build_tool_search_policy
from chatbot.tools.tool_search_uploaded import build_tool_search_uploaded
from chatbot.tools.tool_list_files import tool_list_uploaded_files
from chatbot.llm.llm_text import create_text_llm
from chatbot.router.graph_builder import build_multi_agent_graph


def build_rag_agent(genai_client):
    # 1. Init Components
    text_llm = create_text_llm()
    rag_pipeline = AdvancedRagPipeline(genai_client, text_llm)

    # 2. Build Tools (Chia nhóm)
    # Nhóm Policy
    tool_policy = build_tool_search_policy(rag_pipeline)

    # Nhóm Personal
    tool_uploaded = build_tool_search_uploaded(rag_pipeline, genai_client)

    # 3. Build Multi-Agent Graph
    # Truyền tools riêng biệt cho từng nhóm worker
    app_graph = build_multi_agent_graph(
        text_llm=text_llm,
        tools_policy=[tool_policy],
        tools_personal=[tool_uploaded, tool_list_uploaded_files]
    )

    return app_graph, text_llm
