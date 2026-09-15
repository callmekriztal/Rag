from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.search.RAGSearch") as mock_rag_class:
        mock_instance = MagicMock()
        mock_rag_class.return_value = mock_instance
        from main import app
        with TestClient(app) as test_client:
            yield test_client


def test_healthz(client: TestClient):
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "rag-chat-with-documents"


def test_upload_invalid_extension(client: TestClient):
    response = client.post(
        "/upload",
        files={"file": ("test.exe", b"binary data", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "not allowed" in response.json()["detail"]


def test_query_endpoint(client: TestClient):
    import main
    mock_rag = MagicMock()
    
    async def mock_asearch(query: str, top_k: int = 5):
        return {
            "answer": "Attention is a mechanism in deep learning.",
            "sources": [{"file": "paper.pdf", "page": 1, "sheet": None, "label": "paper.pdf (Page 1)"}],
        }

    mock_rag.asearch_and_answer.side_effect = mock_asearch

    with patch.object(main, "rag", mock_rag):
        response = client.post(
            "/query",
            json={"question": "What is attention?", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is attention?"
        assert "Attention is a mechanism" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file"] == "paper.pdf"
