from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

MULTI_QUERY_PROMPT = """
Bạn là chuyên gia tìm kiếm. 
Hãy tạo ra 3 phiên bản khác nhau của câu hỏi sau để tối ưu tìm kiếm tài liệu (giữ nguyên ý nghĩa, tập trung từ khóa kỹ thuật).
Chỉ trả về 3 dòng câu hỏi, không đánh số.

Câu hỏi gốc: {question}
"""

class QueryGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.chain = (
            PromptTemplate.from_template(MULTI_QUERY_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    def generate_queries(self, original_query: str) -> list[str]:
        try:
            result = self.chain.invoke({"question": original_query})
            variants = [line.strip() for line in result.splitlines() if line.strip()]
            # Luôn giữ query gốc ở đầu
            return [original_query] + variants[:2]
        except Exception as e:
            print(f"[QueryGen] Error: {e}")
            return [original_query]
