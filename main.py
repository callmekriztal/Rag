from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import os
import uuid

rag = None

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".json"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    from src.search import RAGSearch
    rag = RAGSearch()
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/query")
def query(request: QueryRequest):
    answer = rag.search_and_answer(
        query=request.question,
        top_k=request.top_k,
    )
    return {"question": request.question, "answer": answer}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    original_name = file.filename or ""
    ext = Path(original_name).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported")

    # never trust the client-supplied filename as a path component
    safe_name = f"{uuid.uuid4().hex}{ext}"
    save_path = Path("data") / safe_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    size = 0
    with open(save_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):  # 1 MB at a time
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                f.close()
                save_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds max size of {MAX_UPLOAD_SIZE // (1024*1024)}MB",
                )
            f.write(chunk)

    from src.dataloader import load_pdf, load_txt, load_docx, load_csv, load_json

    loaders = {
        ".pdf": load_pdf, ".txt": load_txt,
        ".docx": load_docx, ".csv": load_csv, ".json": load_json
    }

    docs = loaders[ext](save_path)

    # keep the original filename visible in metadata even though we stored it safely
    for doc in docs:
        doc.metadata["original_filename"] = original_name

    rag.vectorstore.add_documents(docs)

    return {"message": f"{original_name} uploaded and indexed", "stored_as": safe_name}