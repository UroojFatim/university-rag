from __future__ import annotations
from typing import TypedDict, List, Dict, Any, NotRequired
import os
import json
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END

from app.config import settings
from app.rag.vectorstore import get_vectorstore
from app.rag.llm import get_llm


_SESSION_HISTORY: Dict[str, List[BaseMessage]] = {}
_HISTORY_DIR = os.path.join(settings.data_dir, "session_history")


class RAGState(TypedDict):
    session_id: str
    question: str
    history: List[BaseMessage]
    docs: NotRequired[List[Document]]
    answer: NotRequired[str]
    sources: NotRequired[List[Dict[str, Any]]]


SYSTEM_PROMPT = """You are a helpful university teaching assistant.
Answer questions primarily using the uploaded slides/notes as your main reference.
You may rephrase, simplify, or lightly explain concepts to help student understanding, but do not introduce information that is not supported by the uploaded materials.

Keep answers concise, clear, and student-friendly, with brief explanations only where helpful.
Do not over-quote or strictly reference slide wording—use the content as prior knowledge instead.

If the answer cannot be derived from the uploaded materials, respond exactly with:
"I don't have that in the uploaded materials."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("system", "Context:\n{context}"),
        ("human", "Question: {question}"),
    ]
)


def _get_history_file(session_id: str) -> str:
    os.makedirs(_HISTORY_DIR, exist_ok=True)
    return os.path.join(_HISTORY_DIR, f"{session_id}.json")


def _serialize_message(msg: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain message to a serializable dict."""
    if isinstance(msg, HumanMessage):
        return {"type": "human", "content": msg.content}
    elif isinstance(msg, AIMessage):
        return {"type": "ai", "content": msg.content}
    else:
        return {"type": "other", "content": str(msg.content)}


def _deserialize_message(data: Dict[str, Any]) -> BaseMessage:
    """Convert a dict back to a LangChain message."""
    msg_type = data.get("type", "other")
    content = data.get("content", "")
    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    else:
        return AIMessage(content=content)


def _load_history_from_disk(session_id: str) -> List[BaseMessage]:
    """Load chat history from disk."""
    history_file = _get_history_file(session_id)
    if not os.path.exists(history_file):
        return []
    
    try:
        with open(history_file, "r") as f:
            data = json.load(f)
        return [_deserialize_message(msg) for msg in data.get("messages", [])]
    except Exception as e:
        print(f"Error loading history for {session_id}: {e}")
        return []


def _save_history_to_disk(session_id: str, history: List[BaseMessage]) -> None:
    """Save chat history to disk."""
    history_file = _get_history_file(session_id)
    try:
        data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "messages": [_serialize_message(msg) for msg in history],
        }
        with open(history_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving history for {session_id}: {e}")


def _get_history(session_id: str) -> List[BaseMessage]:
    # Load from memory cache first
    if session_id in _SESSION_HISTORY:
        return _SESSION_HISTORY[session_id]
    
    # If not in memory, load from disk
    history = _load_history_from_disk(session_id)
    _SESSION_HISTORY[session_id] = history
    return history


def _append_history(session_id: str, msg: BaseMessage) -> None:
    _SESSION_HISTORY.setdefault(session_id, []).append(msg)
    if len(_SESSION_HISTORY[session_id]) > 20:
        _SESSION_HISTORY[session_id] = _SESSION_HISTORY[session_id][-20:]
    
    # Persist to disk
    _save_history_to_disk(session_id, _SESSION_HISTORY[session_id])


def retrieve_node(state: RAGState) -> RAGState:
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": settings.top_k})
    docs = retriever.invoke(state["question"])

    sources = []
    for d in docs:
        meta = d.metadata or {}
        sources.append(
            {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
                "text_preview": (d.page_content[:200] + "…") if len(d.page_content) > 200 else d.page_content,
            }
        )

    return {**state, "docs": docs, "sources": sources}


def generate_node(state: RAGState) -> RAGState:
    llm = get_llm()

    context = "\n\n---\n\n".join(
        [f"[{i+1}] {d.page_content}" for i, d in enumerate(state.get("docs", []))]
    )

    prompt_msgs = PROMPT.format_messages(context=context, question=state["question"])
    resp = llm.invoke(prompt_msgs)
    answer = resp.content if hasattr(resp, "content") else str(resp)

    result: RAGState = {**state, "answer": str(answer)}  # type: ignore
    return result


def memory_node(state: RAGState) -> RAGState:
    sid = state["session_id"]
    _append_history(sid, HumanMessage(content=state["question"]))
    # answer should be present from generate_node
    if "answer" in state:
        _append_history(sid, AIMessage(content=state["answer"]))
    return state


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("memory", memory_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "memory")
    g.add_edge("memory", END)

    return g.compile()


GRAPH = build_graph()


def run_rag(session_id: str, question: str) -> Dict[str, Any]:
    history = _get_history(session_id)
    return GRAPH.invoke({"session_id": session_id, "question": question, "history": history})
