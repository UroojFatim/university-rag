from langchain_groq import ChatGroq
from app.config import settings


def get_llm():
    if not settings.groq_api_key.strip():
        raise ValueError("GROQ_API_KEY is missing in .env")

    # ChatGroq is LangChain’s Groq integration :contentReference[oaicite:4]{index=4}
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
    )
