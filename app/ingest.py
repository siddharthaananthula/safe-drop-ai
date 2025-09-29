import os, glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_store")

def iter_docs():
    patterns = [
        os.path.join(DATA_DIR, "**/*.txt"),
        os.path.join(DATA_DIR, "**/*.md"),
        os.path.join(DATA_DIR, "**/*.pdf"),
    ]
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            yield path

def load_documents():
    docs = []
    for path in iter_docs():
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
    return docs

def main():
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} raw docs")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, emb, persist_directory=PERSIST_DIR)
    # vectordb.persist()
    print(f"Persisted Chroma DB at {PERSIST_DIR}")

if __name__ == "__main__":
    main()
