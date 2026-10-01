# Document RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain**, **Google Gemini**, and **FAISS**. Ingest documents, perform cosine vector similarity search, and generate grounded answers with source citations.

---

## Features

- **Multi-Format Ingestion**: Supports `.pdf`, `.txt`, `.docx`, `.csv`, `.xlsx`, and `.json`.
- **Cosine Vector Search**: L2-normalized embeddings via `models/gemini-embedding-001` and `FAISS IndexFlatIP`.
- **Source Citations**: Answers include exact source document names, page numbers, and sheet labels.
- **Thread-Safe & Non-Blocking**: FAISS operations protected with concurrency locks; async endpoints powered by worker threads (`asyncio.to_thread`).
- **Interactive UI & API**: Web chat frontend included, plus REST endpoints for queries, uploads, and health monitoring.

---

## Quickstart

### 1. Prerequisites
Ensure Python 3.11+ and [`uv`](https://github.com/astral-sh/uv) are installed.

### 2. Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/callmekriztal/Rag.git
cd Rag

# Synchronize dependencies
uv sync

# Create environment configuration
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
```

---

## Running the Application

### Web Application (FastAPI + UI)
```bash
uv run uvicorn main:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser to interact with the chat interface and upload documents.

### CLI Runner
```bash
uv run python app.py
```

### Running Tests
```bash
uv run pytest
```

---

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /healthz` | `GET` | Health check endpoint (`{"status": "ok"}`) |
| `POST /upload` | `POST` | Upload and index a document (`multipart/form-data`) |
| `POST /query` | `POST` | Submit a question (`{"question": "...", "top_k": 3}`) |
| `GET /` | `GET` | Serves the single-page web UI |
