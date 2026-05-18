from chatbot.services.rag_pipeline import AdvancedRagPipeline
from chatbot.tools.tool_search_law import build_tool_search_law
from chatbot.tools.tool_search_uploaded import build_tool_search_uploaded
from chatbot.tools.tool_list_files import tool_list_uploaded_files
from chatbot.tools.tool_recall_history import tool_recall_chat_history
from chatbot.llm.llm_text import create_text_llm
from chatbot.router.graph_builder import build_multi_agent_graph


def build_rag_agent(genai_client, vision_service, *, embedder, vector_store):
    # 1. Init Components
    text_llm = create_text_llm()
    rag_pipeline = AdvancedRagPipeline(
        genai_client=genai_client,
        text_llm_langchain=text_llm,
        embedder=embedder,
        vector_store=vector_store,
    )

    # 2. Build Tools (Chia nhóm)
    tool_law = build_tool_search_law(rag_pipeline)

    # Nhóm Personal
    tool_uploaded = build_tool_search_uploaded(rag_pipeline)

    # 3. Build Multi-Agent Graph
    # Truyền tools riêng biệt cho từng nhóm worker
    app_graph = build_multi_agent_graph(
        text_llm=text_llm,
        tools_policy=[tool_law],
        tools_personal=[tool_uploaded, tool_list_uploaded_files, tool_recall_chat_history],
        vision_service=vision_service
    )

    return app_graph, text_llm
