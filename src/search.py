import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.dataloader import load_all_documents
from src.vectorstore import FaissVectorStore

load_dotenv()
logger = logging.getLogger("new_rag.search")


class RAGSearch:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.PERSIST_DIR
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.llm_model_name = llm_model or settings.LLM_MODEL
        self.data_dir = settings.DATA_DIR

        self.vectorstore = FaissVectorStore(
            persist_dir=self.persist_dir,
            embedding_model=self.embedding_model,
        )

        loaded = self.vectorstore.load()
        if not loaded or self.vectorstore.index is None or self.vectorstore.index.ntotal == 0:
            logger.info("Initializing vector store from existing documents in data directory...")
            docs = load_all_documents(self.data_dir)
            if docs:
                self.vectorstore.build_from_documents(docs)

        google_api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model_name,
            google_api_key=google_api_key if google_api_key else None,
            temperature=0,
        )

        logger.info(f"RAGSearch initialized with LLM: {self.llm_model_name}")

    def search_and_answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        results = self.vectorstore.query(query, top_k=top_k)

        if not results:
            return {
                "answer": "I could not find relevant documents to answer your question.",
                "sources": [],
            }

        context_blocks = []
        sources = []
        seen_sources = set()

        for idx, res in enumerate(results, start=1):
            meta = res.get("metadata") or {}
            text = meta.get("text", "")
            source_file = meta.get("original_filename") or meta.get("source") or "Unknown"
            page = meta.get("page")
            sheet = meta.get("sheet")

            source_label = f"{source_file}"
            if page:
                source_label += f" (Page {page})"
            elif sheet:
                source_label += f" (Sheet: {sheet})"

            context_blocks.append(f"[{idx}] Source: {source_label}\n{text}")

            source_key = (source_file, page, sheet)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "file": source_file,
                    "page": page,
                    "sheet": sheet,
                    "label": source_label,
                })

        context = "\n\n".join(context_blocks)

        prompt = f"""You are a helpful and precise RAG assistant.

Answer the question strictly using the supplied context below.
Include citations (e.g. [1], [2]) corresponding to the context sources when making statements.

If the answer is not present in the context, respond with:
"I could not find that information in the provided documents."

Question:
{query}

Context:
{context}

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            answer_text = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Error invoking Gemini LLM: {e}", exc_info=True)
            answer_text = f"An error occurred while generating the answer: {str(e)}"

        return {
            "answer": answer_text,
            "sources": sources,
        }

    async def asearch_and_answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.search_and_answer, query, top_k)


if __name__ == "__main__":
    rag_search = RAGSearch()
    response = rag_search.search_and_answer("What is attention mechanism?", top_k=3)
    print("\nAnswer:", response["answer"])
    print("Sources:", response["sources"])
