import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA




# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_store")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "policy_aware.txt")


def build_chain(k: int = 3) -> RetrievalQA:
    """
    Build a RetrievalQA chain that:
      - loads the persisted Chroma index
      - retrieves top-k chunks
      - answers with an Ollama local model using a policy-aware prompt
    """
    # 1) embeddings + vector store
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # 2) prompt
    if not os.path.exists(PROMPT_PATH):
        raise FileNotFoundError(f"Prompt not found at: {PROMPT_PATH}")
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    prompt = PromptTemplate.from_template(
        template + "\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    # 3) LLM (Ollama must be installed & running; pull a model e.g. `ollama pull llama3`)
    llm = Ollama(model="llama3:8b")

    # 4) RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


if __name__ == "__main__":
    # quick sanity check
    try:
        chain = build_chain()
        q = "Intercom not working but I'm home—what happens to my parcel?"
        out = chain(q)
        print("Answer:\n", out["result"], "\n")
        print("Sources used:")
        for d in out.get("source_documents", []):
            meta = d.metadata
            src = meta.get("source") or meta.get("file_path") or "unknown"
            print("-", src)
    except Exception as e:
        print("Error while building/running the chain:", repr(e))
        print(
            "\nTips:\n"
            " - Ensure you ran `python app\\ingest.py` first (to create chroma_store/)\n"
            " - Make sure Ollama is installed & running and you pulled a model:\n"
            "     ollama pull llama3\n"
            " - Check the prompt file exists at prompts\\policy_aware.txt\n"
        )
