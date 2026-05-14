"""
CLI smoke test for the self-hosted law store.

Replaces the old Google `file_search_stores` test (which sent the query
to `client.models.generate_content(tools=[FileSearch(...)])`). Now we
exercise the same code path the production agent uses:

    1. embed the query (Cohere)
    2. similarity search in Qdrant (law_corpus)
    3. rerank + evaluator + generate via AdvancedRagPipeline

Usage:

    python -m chatbot.setup_main_store.test_query \\
        "Điều kiện thành lập công ty TNHH một thành viên là gì?"

    # Skip the LLM stage and just print the top-k retrieved chunks
    python -m chatbot.setup_main_store.test_query --retrieve-only \\
        "Cổ đông sáng lập có quyền gì?"
"""
from __future__ import annotations

import argparse
import logging
import sys

from chatbot.core.embedder import CohereEmbedder
from chatbot.core.vectorstore import QdrantVectorStore

logger = logging.getLogger(__name__)


def cmd_retrieve_only(query: str, top_k: int) -> int:
    embedder = CohereEmbedder()
    store = QdrantVectorStore()

    qvec = embedder.embed_query(query)
    hits = store.search(
        collection=store.law_collection,
        query_vector=qvec,
        top_k=top_k,
    )

    if not hits:
        print("(no hits)")
        return 0

    for i, hit in enumerate(hits, 1):
        print(f"--- HIT {i}  score={hit.score:.4f}  [{hit.citation or 'n/a'}] ---")
        print(hit.text[:600])
        print()
    return 0


def cmd_full(query: str) -> int:
    # Late import: building the full pipeline pulls in Gemini, which we
    # don't want to require for --retrieve-only mode.
    import google.genai as genai

    from chatbot.config import config as app_config
    from chatbot.llm.llm_text import create_text_llm
    from chatbot.services.rag_pipeline import AdvancedRagPipeline

    embedder = CohereEmbedder()
    store = QdrantVectorStore()
    text_llm = create_text_llm()
    genai_client = genai.Client(api_key=app_config.GOOGLE_API_KEY)

    pipeline = AdvancedRagPipeline(
        genai_client=genai_client,
        text_llm_langchain=text_llm,
        embedder=embedder,
        vector_store=store,
    )
    answer = pipeline.run_pipeline(
        original_query=query,
        collection=store.law_collection,
    )

    print("=" * 50)
    print("ANSWER")
    print("=" * 50)
    print(answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="Câu hỏi cần test")
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Skip the LLM generate stage; just show top-k retrieved chunks.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Top-k for --retrieve-only mode."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    query = " ".join(args.query)
    if args.retrieve_only:
        return cmd_retrieve_only(query, args.top_k)
    return cmd_full(query)


if __name__ == "__main__":
    sys.exit(main())
