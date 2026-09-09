from langchain.tools import tool
#from duckduckgo_search import DDGS
from ddgs import DDGS
from app.rag.retriever import answer_from_docs
from app.rag.vector_store import load_vector_store

@tool
def document_qa_tool(question: str) -> str:
    """Answer questions using the uploaded document(s). Use this for anything
    that could be in the user's uploaded PDF — definitions, explanations,
    specific content from the document."""
    vectordb = load_vector_store()
    answer, sources = answer_from_docs(vectordb, question)
    return f"{answer}\n(Sources: pages {sources})"

@tool
def web_search_tool(query: str) -> str:
    """Search the web for current or general information NOT likely to be
    in the uploaded document — news, current events, facts outside the doc."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])

@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '2 + 2 * 3'."""
    try:
        # Safe eval: only allow numbers and basic operators
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Invalid expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"