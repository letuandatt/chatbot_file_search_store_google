import google.genai.types as types
from chatbot.core.reranker import CohereReranker
from chatbot.core.query_generator import QueryGenerator
from chatbot.core.evaluator import RelevanceEvaluator
from chatbot.config import config as app_config


class AdvancedRagPipeline:
    def __init__(self, genai_client, text_llm_langchain):
        self.client = genai_client
        self.reranker = CohereReranker()
        self.query_gen = QueryGenerator(text_llm_langchain)
        self.evaluator = RelevanceEvaluator(text_llm_langchain)
        self.model_name = app_config.TEXT_MODEL_NAME

    def _fetch_chunks(self, query: str, store_names: list[str]) -> list[str]:
        """Helper: Gọi Google lấy chunks"""
        try:
            tool_config = types.Tool(
                file_search=types.FileSearch(file_search_store_names=store_names)
            )
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
        except Exception:
            return []

    def run_pipeline(self, original_query: str, store_names: list[str]) -> str:
        # 1. Sinh các biến thể câu hỏi (Multi-query)
        queries = self.query_gen.generate_queries(original_query)
        print(f"[Pipeline] Generated queries: {queries}")

        final_relevant_chunks = []

        # 2. Vòng lặp CRAG (Corrective Loop)
        # Thử từng query một, nếu tìm được tài liệu ngon (Evaluator say YES) thì dừng sớm để tiết kiệm.
        for q in queries:
            print(f"--- Trying query: {q} ---")

            # A. Search
            raw_chunks = self._fetch_chunks(q, store_names)
            if not raw_chunks: continue

            # B. Rerank (Lọc sơ bộ bằng Cohere trước)
            top_chunks = self.reranker.rerank(q, list(set(raw_chunks)), top_n=3)

            # C. Evaluation (Chấm điểm kỹ bằng LLM)
            good_chunks_in_pass = []
            for chunk in top_chunks:
                grade = self.evaluator.evaluate(original_query, chunk)
                if grade == "YES":
                    good_chunks_in_pass.append(chunk)
                else:
                    print(f"[Evaluator] Rejected a chunk for query '{q}'")

            # D. Decision (Quyết định)
            if good_chunks_in_pass:
                print(f"[Pipeline] Found {len(good_chunks_in_pass)} good chunks with query '{q}'.")
                final_relevant_chunks.extend(good_chunks_in_pass)
                # Nếu đã tìm thấy ít nhất 2 đoạn ngon, có thể dừng tìm kiếm để trả lời cho nhanh
                if len(final_relevant_chunks) >= 2:
                    break
            else:
                print(f"[Pipeline] Query '{q}' yielded no relevant info. Retrying next variant...")

        # 3. Tổng hợp kết quả
        if not final_relevant_chunks:
            # Fallback: Nếu lục tung cả 3 câu hỏi mà Evaluator vẫn say NO hết
            return "Xin lỗi, tôi đã thử tìm kiếm trong tài liệu nhưng không thấy thông tin liên quan đến câu hỏi của bạn. (CRAG: No relevant docs found)"

        # Deduplicate lần cuối
        unique_context = list(set(final_relevant_chunks))
        context_text = "\n\n---\n\n".join(unique_context)

        # 4. Generate Answer
        final_prompt = f"""Dựa vào các thông tin ĐÃ ĐƯỢC KIỂM CHỨNG sau đây, hãy trả lời câu hỏi.

NGỮ CẢNH (CONTEXT):
{context_text}

CÂU HỎI:
{original_query}
"""
        try:
            res = self.client.models.generate_content(model=self.model_name, contents=final_prompt)
            return res.text
        except Exception as e:
            return f"Lỗi tổng hợp: {e}"