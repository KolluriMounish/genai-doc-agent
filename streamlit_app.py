import os
import streamlit as st
from app.ingestion.loader import load_and_chunk_pdf
from app.rag.vector_store import build_vector_store
from app.agent.graph import run_agent
import shutil

st.set_page_config(page_title="Enterprise Doc & Data Assistant", page_icon="🤖", layout="centered")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- Session state setup ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_ready" not in st.session_state:
    st.session_state.doc_ready = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ---- Sidebar: document upload ----
with st.sidebar:
    st.header("📄 Upload a document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        # if st.button("Process document"):
        #     with st.spinner("Reading, chunking, and embedding your document..."):
        #         file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        #         with open(file_path, "wb") as f:
        #             f.write(uploaded_file.getbuffer())

        #         chunks = load_and_chunk_pdf(file_path)
        #         build_vector_store(chunks)

        #         st.session_state.doc_ready = True
        #         st.session_state.doc_name = uploaded_file.name

        #     st.success(f"'{uploaded_file.name}' processed — {len(chunks)} chunks indexed.")

        if st.button("Process document"):
            with st.spinner("Reading, chunking, and embedding your document..."):
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db")

                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                chunks = load_and_chunk_pdf(file_path)
                build_vector_store(chunks)

                st.session_state.doc_ready = True
                st.session_state.doc_name = uploaded_file.name

            st.success(f"'{uploaded_file.name}' processed — {len(chunks)} chunks indexed.")

    if st.session_state.doc_ready:
        st.info(f"Active document: **{st.session_state.doc_name}**")

    st.divider()
    st.caption(
        "This assistant can answer from your uploaded document, "
        "search the web for current info, or do quick calculations — "
        "it decides automatically."
    )

# ---- Main chat area ----
st.title("🤖 Enterprise Doc & Data Assistant")
st.caption("RAG + Agentic tool-use, powered by Groq (Llama/GPT-OSS), LangGraph & ChromaDB")

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            st.caption(f"🔧 Tool(s) used: {', '.join(msg['tools_used'])}")

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, tools_used = run_agent(user_input)
            st.markdown(answer)
            if tools_used:
                st.caption(f"🔧 Tool(s) used: {', '.join(tools_used)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tools_used": tools_used,
    })

if st.button("Process document"):
    with st.spinner("Reading, chunking, and embedding your document..."):
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        chunks = load_and_chunk_pdf(file_path)
        build_vector_store(chunks)

        st.session_state.doc_ready = True
        st.session_state.doc_name = uploaded_file.name

    st.success(f"'{uploaded_file.name}' processed — {len(chunks)} chunks indexed.")