from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = Field(default="University RAG", alias="APP_NAME")

    data_dir: str = Field(default="./data", alias="DATA_DIR")
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    chroma_dir: str = Field(default="./data/chroma", alias="CHROMA_DIR")

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )

    top_k: int = Field(default=6, alias="TOP_K")

    # Groq
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.2, alias="GROQ_TEMPERATURE")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
