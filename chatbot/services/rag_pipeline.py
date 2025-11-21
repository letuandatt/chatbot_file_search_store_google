import google.genai.types as types
from chatbot.core.reranker import CohereReranker
from chatbot.core.query_generator import QueryGenerator
from chatbot.config import config as app_config


class AdvancedRagPipeline:
    def __init__(self, genai_client, text_llm_langchain):
        self.client = genai_client
        self.reranker = CohereReranker()
        self.query_gen = QueryGenerator(text_llm_langchain)
        self.model_name = app_config.TEXT_MODEL_NAME

    def _fetch_chunks(self, query: str, store_names: list[str]) -> list[str]:
        """Gọi Google để lấy chunks thô, không lấy câu trả lời."""
        try:
            tool_config = types.Tool(
                file_search=types.FileSearch(file_search_store_names=store_names)
            )
            # Prompt trick để ép Google trả về chunks mà không cần generate text quá nhiều
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Trích xuất thông tin liên quan đến: {query}",
                config=types.GenerateContentConfig(tools=[tool_config])
            )

            chunks = []
            if hasattr(response, 'candidates') and response.candidates:
                cand = response.candidates[0]
                if cand.grounding_metadata and cand.grounding_metadata.grounding_chunks:
                    for chunk in cand.grounding_metadata.grounding_chunks:
                        if hasattr(chunk, 'retrieved_context'):
                            chunks.append(chunk.retrieved_context.text)
            return chunks
        except Exception as e:
            print(f"[Pipeline] Search Error: {e}")
            return []

    def run_pipeline(self, original_query: str, store_names: list[str]) -> str:
        # 1. Multi-Query
        queries = self.query_gen.generate_queries(original_query)
        print(f"[Pipeline] Queries: {queries}")

        # 2. Search (Parallel/Sequential)
        all_chunks = []
        for q in queries:
            all_chunks.extend(self._fetch_chunks(q, store_names))

        unique_chunks = list(set(all_chunks))
        if not unique_chunks:
            return "Không tìm thấy thông tin trong tài liệu."

        # 3. Rerank
        top_chunks = self.reranker.rerank(original_query, unique_chunks, top_n=5)

        # 4. Generate Answer
        context = "\n\n---\n\n".join(top_chunks)
        final_prompt = f"""Dựa vào ngữ cảnh sau, trả lời câu hỏi chi tiết.
Nếu không có thông tin, hãy nói không biết.

NGỮ CẢNH:
{context}

CÂU HỎI:
{original_query}
"""
        try:
            res = self.client.models.generate_content(
                model=self.model_name,
                contents=final_prompt
            )
            return res.text
        except Exception as e:
            return f"Lỗi tổng hợp: {e}"