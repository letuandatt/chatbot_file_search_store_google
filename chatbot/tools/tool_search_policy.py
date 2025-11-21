from langchain_core.tools import StructuredTool
from chatbot.core.cache import app_cache
from chatbot.config import config as app_config


def build_tool_search_policy(rag_pipeline):
    """Factory function: Tạo tool và inject genai_client vào scope."""

    def search_policy_logic(query: str):
        """
        Dùng để tra cứu các QUY ĐỊNH CHUNG, QUY TRÌNH, THỦ TỤC, BIỂU MẪU, HƯỚNG DẪN của tổ chức (CUSC).
        """
        if not app_config.CUSC_MAIN_STORE_NAME:
            return "Hệ thống chưa được cấu hình Main Store."

        cache_k = app_cache.generate_key("policy", "adv", query)
        cached = app_cache.get(cache_k)
        if cached:
            return cached

        try:
            result = rag_pipeline.run_pipeline(
                original_query=query,
                store_names=[app_config.CUSC_MAIN_STORE_NAME]
            )

            app_cache.set(cache_k, result, ttl=3600)
            return result
        except Exception as e:
            return f"Lỗi khi tra cứu quy trình chung: {e}"

    return StructuredTool.from_function(
        func=search_policy_logic,
        name="tool_search_general_policy",
        description="Tra cứu quy định chung, quy trình, biểu mẫu, ISO."
    )
