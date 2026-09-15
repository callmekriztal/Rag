import numpy as np
import pytest
from pathlib import Path
from src.vectorstore import FaissVectorStore
from src.embedding import normalize_embeddings


def test_normalize_embeddings():
    vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normed = normalize_embeddings(vecs)
    assert pytest.approx(np.linalg.norm(normed[0])) == 1.0
    assert pytest.approx(np.linalg.norm(normed[1])) == 0.0


def test_faiss_vector_store_add_and_search(tmp_path: Path):
    store = FaissVectorStore(persist_dir=str(tmp_path))

    # Synthetic normalized vectors (dim 4)
    vec1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vec2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    embeddings = np.vstack([vec1, vec2])

    metadatas = [
        {"text": "Document 1 content", "source": "doc1.txt"},
        {"text": "Document 2 content", "source": "doc2.txt"},
    ]

    store.add_embeddings(embeddings, metadatas)
    assert store.index.ntotal == 2

    # Query with vector close to vec1
    query_vec = np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32)
    query_vec = normalize_embeddings(query_vec)

    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0]["metadata"]["source"] == "doc1.txt"

    # Save and reload
    store.save()

    new_store = FaissVectorStore(persist_dir=str(tmp_path))
    loaded = new_store.load()
    assert loaded is True
    assert new_store.index.ntotal == 2
    assert new_store.metadata[0]["source"] == "doc1.txt"
