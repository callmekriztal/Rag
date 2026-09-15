import logging
from typing import List
import numpy as np
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings

logger = logging.getLogger("new_rag.embedding")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings for cosine similarity searching."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Prevent division by zero
    return (embeddings / norms).astype(np.float32)


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self.model = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=settings.GOOGLE_API_KEY if settings.GOOGLE_API_KEY else None,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: List[Document]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        if not texts:
            return np.empty((0, 768), dtype=np.float32)
        raw_embeddings = self.model.embed_documents(texts)
        arr = np.array(raw_embeddings, dtype=np.float32)
        return normalize_embeddings(arr)

    def embed_query(self, query: str) -> np.ndarray:
        raw_embedding = self.model.embed_query(query)
        arr = np.array([raw_embedding], dtype=np.float32)
        return normalize_embeddings(arr)
