from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SourceChunk(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    text_preview: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceChunk]
    debug: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
