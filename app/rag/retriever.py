import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_groq_key()

load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    #model_name="llama-3.3-70b-versatile",
    temperature=0.1,
)
print("Using model:", llm.model_name)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions using ONLY the provided context.
If the answer isn't in the context, say you don't know — don't make it up.

Context:
{context}

Question: {question}

Answer:
""")

def answer_from_docs(vectordb, question: str, k: int = 4):
    """Retrieves top-k relevant chunks and generates an answer."""
    results = vectordb.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in results])

    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    sources = [doc.metadata.get("page", "unknown") for doc in results]
    return response.content, sources