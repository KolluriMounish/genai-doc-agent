from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path: str):
    """
    Loads a PDF and splits it into overlapping chunks.
    chunk_size: how many characters per chunk
    chunk_overlap: characters shared between consecutive chunks,
                   so we don't lose context at chunk boundaries
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)
    return chunks