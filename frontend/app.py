import streamlit as st
import requests
from typing import List, Dict, Any

st.set_page_config(page_title="University RAG Chatbot", page_icon="🎓", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def ingest_file(api_base: str, file) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/ingest"
    files = {"file": (file.name, file.getvalue())}
    r = requests.post(url, files=files, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"Ingest failed: {r.status_code} - {r.text}")
    return r.json()


def chat(api_base: str, session_id: str, message: str) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/chat"
    payload = {"session_id": session_id, "message": message}
    r = requests.post(url, json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"Chat failed: {r.status_code} - {r.text}")
    return r.json()


def health(api_base: str) -> bool:
    url = f"{api_base.rstrip('/')}/health"
    try:
        r = requests.get(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def get_ingested_files(api_base: str) -> List[Dict[str, Any]]:
    """Get list of already-ingested files from backend"""
    url = f"{api_base.rstrip('/')}/ingested-files"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("files", [])
    except Exception:
        pass
    return []


def render_sources(sources: List[Dict[str, Any]]):
    if not sources:
        st.caption("No sources returned.")
        return

    with st.expander("Sources (evidence from your slides/notes)", expanded=False):
        for i, s in enumerate(sources, start=1):
            src = s.get("source", "unknown")
            page = s.get("page")
            chunk_id = s.get("chunk_id")
            preview = s.get("text_preview", "")

            header = f"{i}. {src}"
            if page is not None:
                header += f" — page/slide {page}"

            st.markdown(f"**{header}**")
            if chunk_id:
                st.caption(f"chunk_id: {chunk_id}")
            st.write(preview)
            st.divider()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎓 University RAG Settings")

default_api = st.sidebar.text_input(
    "FastAPI backend URL",
    value=st.session_state.get("api_base", "http://localhost:8000"),
    help="Example: http://localhost:8000",
)
st.session_state["api_base"] = default_api

colA, colB = st.sidebar.columns(2)
with colA:
    check = st.button("✅ Check API")
with colB:
    clear = st.button("🧹 Clear Chat")

if check:
    if default_api:
        ok = health(default_api)
        st.sidebar.success("Backend reachable ✅" if ok else "Backend not reachable ❌")
    else:
        st.sidebar.error("Please provide a valid API URL")

# session id
session_id = st.sidebar.text_input(
    "Session ID",
    value=st.session_state.get("session_id", "uni1"),
    help="Use a unique ID per user/course if you want separate chat memory.",
)
st.session_state["session_id"] = session_id

# Upload + Ingest
st.sidebar.subheader("Upload & Ingest")
uploaded = st.sidebar.file_uploader(
    "Upload PDF/PPTX",
    type=["pdf", "pptx"],
    accept_multiple_files=True,
)

ingest_btn = st.sidebar.button("📥 Ingest Uploaded Files", disabled=(not uploaded))

# Show already-ingested files
ingested = get_ingested_files(default_api) if default_api else []
if ingested:
    st.sidebar.subheader("✅ Already Ingested")
    for f in ingested:
        st.sidebar.caption(f"📄 {f['name']}")
else:
    st.sidebar.caption("*No documents ingested yet*")

if clear:
    st.session_state["messages"] = []
    st.session_state["last_sources"] = []
    st.toast("Chat cleared")

# -----------------------------
# Main UI
# -----------------------------
st.title("🎓 University RAG Chatbot")
st.caption("Upload your lecture slides/notes (PDF/PPTX), then ask questions grounded in your materials.")

# Initialize chat store
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # list of dicts: {role, content}
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = []

# Ingest action
if ingest_btn:
    if not default_api:
        st.error("Please provide a valid API URL")
    elif not health(default_api):
        st.error("Backend is not reachable. Start FastAPI server first.")
    else:
        progress = st.progress(0)
        results = []
        try:
            total = len(uploaded)
            for idx, f in enumerate(uploaded, start=1):
                res = ingest_file(default_api, f)
                results.append(res)
                progress.progress(int(idx / total * 100))
            st.success("✅ Ingestion complete")
            # show summary
            for r in results:
                st.write(f"• **{r['filename']}** → chunks added: **{r['chunks_added']}**")
        except Exception as e:
            st.error(str(e))
        finally:
            progress.empty()

st.divider()

# Render conversation
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("Ask about your lectures, slides, or notes...")

if prompt:
    if not default_api or not session_id:
        st.error("Please provide valid API URL and Session ID")
    elif not health(default_api):
        st.error("Backend is not reachable. Start FastAPI server first.")
    else:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = chat(default_api, session_id, prompt)
                    answer = resp.get("answer", "")
                    sources = resp.get("sources", [])

                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.session_state["last_sources"] = sources

                    render_sources(sources)
                except Exception as e:
                    st.error(str(e))

# Sticky area to show last sources even after scrolling
if st.session_state.get("last_sources"):
    # st.divider()
    # st.subheader("Last answer sources")
    render_sources(st.session_state["last_sources"])
