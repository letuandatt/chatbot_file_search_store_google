import cohere
from chatbot.config import config as app_config


class CohereReranker:
    def __init__(self):
        self.api_key = app_config.COHERE_API_KEY
        self.model = app_config.COHERE_MODEL_NAME
        self.client = None

        if self.api_key:
            try:
                self.client = cohere.ClientV2(api_key=self.api_key)
                print(f"[Reranker] Cohere initialized with model: {self.model}")
            except Exception as e:
                print(f"[Reranker] Failed to init Cohere: {e}")
        else:
            print("[Reranker] Warning: COHERE_API_KEY missing.")

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[str]:
        """
        Input: Query và danh sách document strings.
        Output: Danh sách document strings đã được lọc và sắp xếp.
        """
        if not self.client or not documents:
            # Fallback: Trả về danh sách gốc cắt ngắn nếu không có Reranker
            return documents[:top_n]

        try:
            # Lọc bỏ văn bản rỗng/ngắn
            valid_docs = [d for d in documents if d and len(str(d).strip()) > 10]
            if not valid_docs:
                return []

            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=valid_docs,
                top_n=top_n,
            )

            ranked_texts = []
            for result in response.results:
                ranked_texts.append(valid_docs[result.index])

            return ranked_texts
        except Exception as e:
            print(f"[Reranker] Error: {e}")
            return documents[:top_n]