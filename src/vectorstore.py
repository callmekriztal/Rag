import os
import pickle
from typing import List

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "text-embedding-004",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Vector store initialized with model: {embedding_model}")

    def build_from_documents(self, documents: List[Document]):
        print(f"[INFO] Building vector store from {len(documents)} documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)

        metadatas = [
            {"text": chunk.page_content, **chunk.metadata}
            for chunk in chunks
        ]

        self.add_embeddings(embeddings, metadatas)
        self.save()

        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[dict] = None):
        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(faiss_path):
            print("[WARNING] FAISS index not found.")
            return

        if not os.path.exists(meta_path):
            print("[WARNING] Metadata file not found.")
            return

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({
                "index": int(idx),
                "distance": float(dist),
                "metadata": meta,
            })

        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")

        model = GoogleGenerativeAIEmbeddings(model=self.embedding_model)
        query_emb = np.array(model.embed_query(query_text)).astype(np.float32).reshape(1, -1)

        return self.search(query_emb, top_k=top_k)
