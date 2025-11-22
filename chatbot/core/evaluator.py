from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt chấm điểm cực kỳ khắt khe
EVALUATOR_PROMPT = """Bạn là một người chấm điểm tính liên quan của tài liệu (Relevance Grader).
Nhiệm vụ: Đánh giá xem đoạn văn bản (Document) có chứa thông tin để trả lời câu hỏi (Question) hay không.

- Nếu đoạn văn bản chứa từ khóa hoặc ý nghĩa trả lời được câu hỏi -> Trả về "YES"
- Nếu đoạn văn bản không liên quan hoặc chỉ nói chung chung -> Trả về "NO"

Chỉ trả về duy nhất một từ: "YES" hoặc "NO".

Document: {document}
Question: {question}
"""

class RelevanceEvaluator:
    def __init__(self, llm):
        self.llm = llm
        self.chain = (
            PromptTemplate.from_template(EVALUATOR_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    def evaluate(self, query: str, document: str) -> str:
        """
        Trả về 'YES' hoặc 'NO'.
        """
        try:
            # Gọi LLM để chấm điểm
            score = self.chain.invoke({"question": query, "document": document})
            return score.strip().upper()
        except Exception as e:
            print(f"[Evaluator] Error: {e}")
            # Fallback: Nếu lỗi thì tạm chấp nhận (An toàn)
            return "YES"
