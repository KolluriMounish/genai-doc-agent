from app.ingestion.loader import load_and_chunk_pdf
from app.rag.vector_store import build_vector_store
from app.rag.retriever import answer_from_docs

# Change this to your actual PDF filename
chunks = load_and_chunk_pdf("data/uploads/Maths for ML Revision.pdf")
print(f"Loaded {len(chunks)} chunks")

vectordb = build_vector_store(chunks)

question = "What is this document about?"
answer, sources = answer_from_docs(vectordb, question)

print("\nQuestion:", question)
print("Answer:", answer)
print("Sources (pages):", sources)