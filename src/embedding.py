from typing import List
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "models/gemini-embedding-001",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model = GoogleGenerativeAIEmbeddings(model=model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def embed_chunks(self, chunks: List[Document]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.embed_documents(texts)
        return np.array(embeddings).astype(np.float32)
