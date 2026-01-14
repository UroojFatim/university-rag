import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.rag.ingest import ingest_file
from app.rag.graph import run_rag
from app.rag.schemas import ChatRequest, ChatResponse, IngestResponse, SourceChunk


def ensure_dirs():
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_dir, exist_ok=True)


ensure_dirs()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@app.get("/ingested-files")
def get_ingested_files():
    """Get list of files that have been ingested into the vector store"""
    upload_dir = settings.upload_dir
    if not os.path.exists(upload_dir):
        return {"files": []}
    
    files = []
    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        if os.path.isfile(filepath):
            # Get file size and modification time
            stat = os.stat(filepath)
            files.append({
                "name": filename,
                "size": stat.st_size,
                "ingested_at": stat.st_mtime
            })
    
    return {"files": sorted(files, key=lambda x: x["ingested_at"], reverse=True)}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.strip():
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(filename)[1].lower()
    allowed = {".pdf", ".pptx"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext} (use PDF/PPTX)")

    save_path = os.path.join(settings.upload_dir, filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    try:
        chunks = ingest_file(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")

    return IngestResponse(filename=filename, chunks_added=chunks)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if not payload.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    try:
        out = run_rag(session_id=payload.session_id, question=payload.message)
        sources = [SourceChunk(**s) for s in out.get("sources", [])]
        return ChatResponse(
            session_id=payload.session_id,
            answer=out.get("answer", ""),
            sources=sources,
            debug=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
