import logging
import os
import pickle
import threading
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document

from src.config import get_settings
from src.embedding import EmbeddingPipeline

logger = logging.getLogger("new_rag.vectorstore")


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.PERSIST_DIR
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        os.makedirs(self.persist_dir, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        self.emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        logger.info(f"FaissVectorStore initialized with model: {self.embedding_model}")

    def build_from_documents(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents provided to build vector store.")
            return

        logger.info(f"Building vector store from {len(documents)} documents...")
        chunks = self.emb_pipe.chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks produced from documents.")
            return

        embeddings = self.emb_pipe.embed_chunks(chunks)

        metadatas = [
            {"text": chunk.page_content, **chunk.metadata}
            for chunk in chunks
        ]

        with self._lock:
            self.index = None
            self.metadata = []
            self._add_embeddings_unlocked(embeddings, metadatas)
            self._save_unlocked()

        logger.info(f"Vector store built and saved to {self.persist_dir}")

    def add_documents(self, documents: List[Document]):
        if not documents:
            return

        chunks = self.emb_pipe.chunk_documents(documents)
        if not chunks:
            return

        embeddings = self.emb_pipe.embed_chunks(chunks)
        metadatas = [
            {"text": chunk.page_content, **chunk.metadata}
            for chunk in chunks
        ]

        with self._lock:
            self._add_embeddings_unlocked(embeddings, metadatas)
            self._save_unlocked()

        logger.info(f"Incrementally added {len(chunks)} chunks to vector store.")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: Optional[List[dict]] = None):
        with self._lock:
            self._add_embeddings_unlocked(embeddings, metadatas)

    def _add_embeddings_unlocked(self, embeddings: np.ndarray, metadatas: Optional[List[dict]] = None):
        if embeddings.shape[0] == 0:
            return

        dim = embeddings.shape[1]
        if self.index is None:
            # Use IndexFlatIP for Cosine Similarity (with normalized vectors)
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        logger.info(f"Added {embeddings.shape[0]} vectors to FAISS index.")

    def save(self):
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self):
        if self.index is None:
            logger.warning("Cannot save empty FAISS index.")
            return

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):
        with self._lock:
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            meta_path = os.path.join(self.persist_dir, "metadata.pkl")

            if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
                logger.warning(f"Index or metadata file missing in {self.persist_dir}")
                return False

            self.index = faiss.read_index(faiss_path)

            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

            logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors) and metadata from {self.persist_dir}")
            return True

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("Search called on empty vector store index.")
                return []

            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)

            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                results.append({
                    "index": int(idx),
                    "score": float(score),
                    "metadata": meta,
                })

            return results

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Querying vector store for: '{query_text}'")
        query_emb = self.emb_pipe.embed_query(query_text)
        return self.search(query_emb, top_k=top_k)