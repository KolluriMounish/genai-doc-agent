import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import streamlit as st
from app.rag.retriever import get_groq_key

from app.agent.tools import document_qa_tool, web_search_tool, calculator_tool

load_dotenv()

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_groq_key()

llm = ChatGroq(
    groq_api_key=get_groq_key(),
    model_name="openai/gpt-oss-20b",
    temperature=0.1,
)

tools = [document_qa_tool, web_search_tool, calculator_tool]

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
- Use document_qa_tool for questions about the uploaded document.
- Use web_search_tool for current events or general knowledge questions.
- Use calculator_tool for math.
Always pick the most relevant tool. If unsure, prefer document_qa_tool first."""

#agent = create_react_agent(llm, tools, state_modifier=SYSTEM_PROMPT)
agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

def run_agent(question: str):
    """Runs the agent and returns the final answer plus which tool(s) were used."""
    result = agent.invoke({"messages": [("user", question)]})
    messages = result["messages"]

    final_answer = messages[-1].content

    tools_used = []
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            tools_used.extend([tc["name"] for tc in m.tool_calls])

    return final_answer, tools_used