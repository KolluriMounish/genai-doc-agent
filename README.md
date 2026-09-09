# Enterprise Doc & Data Assistant

A GenAI application combining **RAG (Retrieval-Augmented Generation)** with an
**agentic tool-routing system**, built with LangChain, LangGraph, ChromaDB, and Groq.

## What it does
- Upload a PDF and ask questions grounded in its content (RAG)
- The agent automatically decides whether to answer from the document,
  search the web, or run a calculation
- Built with a ReAct agent pattern (LangGraph) for tool orchestration

## Tech stack
- **LLM**: Groq (openai/gpt-oss-20b)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: ChromaDB
- **Agent framework**: LangChain + LangGraph (ReAct agent)
- **UI**: Streamlit
- **Tools**: Document Q&A, Web Search (DuckDuckGo), Calculator

## Live demo
[link once deployed]

## Run locally
\`\`\`bash
pip install -r requirements.txt
# add GROQ_API_KEY to .env
streamlit run streamlit_app.py
\`\`\`