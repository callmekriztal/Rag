import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "text-embedding-004",
        llm_model: str = "gemini-2.5-flash",
    ):
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model,
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index",
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl",
        )

        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):
            from src.dataloader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        google_api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=google_api_key,
            temperature=0,
        )

        print(
            f"[INFO] Gemini LLM initialized: {llm_model}"
        )

    def search_and_answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:

        results = self.vectorstore.query(
            query,
            top_k=top_k,
        )

        contexts = []

        for result in results:
            metadata = result.get("metadata")

            if metadata:
                contexts.append(
                    metadata.get("text", "")
                )

        context = "\n\n".join(contexts)

        if not context.strip():
            return "No relevant documents found."

        prompt = f"""
You are a helpful RAG assistant.

Answer ONLY using the supplied context.

If the answer is not present in the context, say:

"I could not find that information in the provided documents."

Question:
{query}

Context:
{context}

Answer:
"""

        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":
    rag_search = RAGSearch()

    response = rag_search.search_and_answer(
        "What is attention mechanism?",
        top_k=3,
    )

    print(response)
