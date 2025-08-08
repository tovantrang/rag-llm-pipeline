# 1_ingest.py
import glob
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

import config


def ingest_data():
    """
    Reads all PDF documents from the data path, processes them,
    and stores their embeddings in a FAISS vector store.
    """
    # 1. Load Documents
    pdf_files = glob.glob(os.path.join(config.PDF_DATA_PATH, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in the data directory. Aborting.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    documents = []
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        loader = PyMuPDFLoader(pdf_file)
        documents.extend(loader.load())

    # 2. Chunk Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunked_docs)} chunks.")

    # 3. Create Embeddings
    # Using a local, open-source embedding model
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": config.DEVICE}
    )

    # 4. Create and Save Vector Store
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunked_docs, embeddings)

    # Save the vector store locally
    vector_store.save_local(config.VECTOR_STORE_PATH)
    print(f"Vector store created and saved to {config.VECTOR_STORE_PATH}")


if __name__ == "__main__":
    ingest_data()
