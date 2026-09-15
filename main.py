import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import get_settings
from src.dataloader import load_single_file
from src.search import RAGSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("new_rag.api")

rag: Optional[RAGSearch] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    logger.info("Initializing RAG backend service...")
    rag = RAGSearch()
    logger.info("RAG service successfully initialized.")
    yield


app = FastAPI(
    title="Document RAG API",
    description="Production-ready Retrieval-Augmented Generation API with Gemini and FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The query question for the documents")
    top_k: int = Field(default=3, ge=1, le=20, description="Top K relevant chunks to retrieve")


class SourceItem(BaseModel):
    file: str
    page: Optional[int] = None
    sheet: Optional[str] = None
    label: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]


class UploadResponse(BaseModel):
    message: str
    stored_as: str
    chunks_indexed: int


@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok", "service": "rag-chat-with-documents"}


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is still initializing",
        )

    try:
        result = await rag.asearch_and_answer(
            query=request.question,
            top_k=request.top_k,
        )
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result.get("sources", []),
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@app.post("/upload", response_model=UploadResponse)
async def upload_endpoint(file: UploadFile = File(...)):
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service is still initializing",
        )

    settings = get_settings()
    original_name = file.filename or "uploaded_file"
    ext = Path(original_name).suffix.lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File extension '{ext}' is not allowed. Supported extensions: {settings.ALLOWED_EXTENSIONS}",
        )

    safe_name = f"{uuid.uuid4().hex}{ext}"
    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / safe_name

    size = 0
    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunk
                size += len(chunk)
                if size > settings.MAX_UPLOAD_SIZE:
                    f.close()
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed upload size of {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.error(f"Failed to save upload file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file",
        )

    # Process and index document in worker thread
    def process_and_index():
        docs = load_single_file(save_path)
        for doc in docs:
            doc.metadata["original_filename"] = original_name

        rag.vectorstore.add_documents(docs)
        return len(docs)

    try:
        indexed_count = await asyncio.to_thread(process_and_index)
        return UploadResponse(
            message=f"File '{original_name}' successfully processed and indexed.",
            stored_as=safe_name,
            chunks_indexed=indexed_count,
        )
    except Exception as e:
        logger.error(f"Error parsing and indexing uploaded file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index file: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )