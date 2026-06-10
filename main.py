from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from src.search import RAGSearch

app = FastAPI()
rag = RAGSearch()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def root():
    return {"status": "RAG API is running"}

@app.post("/query")
def query(request: QueryRequest):
    answer = rag.search_and_answer(
        query=request.question,
        top_k=request.top_k,
    )
    return {"question": request.question, "answer": answer}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    allowed = [".pdf", ".txt", ".docx", ".csv", ".json"]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported")
    
    save_path = os.path.join("data", file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    from src.dataloader import load_all_documents
    docs = load_all_documents("data")
    rag.vectorstore.build_from_documents(docs)
    
    return {"message": f"{file.filename} uploaded and indexed"}