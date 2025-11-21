from chatbot.services.rag_pipeline import AdvancedRagPipeline
from chatbot.tools.tool_search_policy import build_tool_search_policy
from chatbot.tools.tool_search_uploaded import build_tool_search_uploaded
from chatbot.tools.tool_list_files import tool_list_uploaded_files
from chatbot.llm.llm_text import create_text_llm
from chatbot.llm.agent_react import create_agent_executor


def build_rag_agent(genai_client):
    # 1. Init Text LLM (LangChain) dùng cho Query Generator
    text_llm = create_text_llm()

    # 2. Init RAG Pipeline (The Engine)
    rag_pipeline = AdvancedRagPipeline(genai_client, text_llm)

    # 3. Build Tools (Inject Pipeline)
    tool_policy = build_tool_search_policy(rag_pipeline)
    tool_uploaded = build_tool_search_uploaded(rag_pipeline, genai_client)

    tools = [tool_policy, tool_uploaded, tool_list_uploaded_files]

    # 4. Create Agent Executor
    agent_executor = create_agent_executor(text_llm, tools)

    return agent_executor, text_llm
