from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_vectordb = None  # holds the current in-memory vector store

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vector_store(chunks):
    """Builds a fresh in-memory vector store, replacing any previous one."""
    global _vectordb
    embeddings = get_embedding_model()
    _vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return _vectordb

def load_vector_store():
    """Returns the currently active vector store."""
    if _vectordb is None:
        raise ValueError("No document has been processed yet. Please upload and process a PDF first.")
    return _vectordb
# #from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# #from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma

# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# PERSIST_DIR = "chroma_db"

# def get_embedding_model():
#     # Runs locally, free, no API calls
#     return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# def build_vector_store(chunks):
#     """Embeds chunks and saves them to disk in ChromaDB."""
#     embeddings = get_embedding_model()
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=PERSIST_DIR,
#     )
#     #vectordb.persist()
#     return vectordb

# def load_vector_store():
#     """Reloads an existing ChromaDB from disk."""
#     embeddings = get_embedding_model()
#     return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)