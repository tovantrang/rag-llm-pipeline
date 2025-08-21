import glob
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from tqdm import tqdm

import config


def ingest_data():
    """
    Reads all PDF documents from the data path, processes them into
    markdown, splits by markdown headers, further chunks within
    sections, and stores embeddings in a FAISS vector store.
    """
    # 1. Load Documents
    pdf_files = glob.glob(os.path.join(config.PDF_DATA_PATH, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in the data directory. Aborting.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    documents = []
    converter = PdfConverter(artifact_dict=create_model_dict())
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            rendered = converter(pdf_file)
            markdown_text, _, _ = text_from_rendered(rendered)
            documents.append(
                Document(page_content=markdown_text, metadata={"source": pdf_file})
            )
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            continue

    # 2. Create Embeddings
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={"device": config.DEVICE}
    )

    # 3. Markdown header-based splitting
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep headers in the chunk text
    )

    all_section_docs = []
    for doc in documents:
        section_docs = md_splitter.split_text(doc.page_content)
        # Carry over the original metadata (PDF source) to each chunk
        for sec_doc in section_docs:
            header_1 = sec_doc.metadata.get("Header 1", "")
            header_2 = sec_doc.metadata.get("Header 2", "")
            sec_doc.metadata["header_1"] = header_1
            sec_doc.metadata["header_2"] = header_2
            sec_doc.metadata["source"] = doc.metadata["source"]

        all_section_docs.extend(section_docs)

    # 4. Further chunk inside each section for embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunked_docs = text_splitter.split_documents(all_section_docs)

    print(f"Split {len(documents)} markdown docs into {len(chunked_docs)} chunks.")

    # 5. Create and Save Vector Store
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    vector_store.save_local(config.VECTOR_STORE_PATH)
    print(f"Vector store created and saved to {config.VECTOR_STORE_PATH}")


if __name__ == "__main__":
    ingest_data()
