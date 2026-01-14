import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings


def get_embeddings():
    # Runs locally (CPU-friendly)
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vectorstore():
    os.makedirs(settings.chroma_dir, exist_ok=True)
    embeddings = get_embeddings()
    return Chroma(
        collection_name="university_rag",
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )
