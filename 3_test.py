import os
from typing import Sequence

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers.cross_encoder import CrossEncoder

import config


class ThresholdReranker(BaseDocumentCompressor):
    """Document compressor that filters retrieved documents using a Cross-Encoder threshold.

    This reranker evaluates the semantic relevance of retrieved documents with
    respect to a query using a Cross-Encoder model. Each (query, document) pair
    is scored, and only documents with a relevance score above the configured
    threshold are retained.

    This component is typically used after vector similarity retrieval (e.g., FAISS)
    to improve retrieval precision by removing false positives and low-relevance
    chunks before passing context to the LLM. This helps reduce noise and improves
    answer quality in Retrieval-Augmented Generation (RAG) pipelines.

    Attributes:
        model (CrossEncoder): Cross-Encoder model used to compute relevance scores
            between the query and each retrieved document.
        threshold (float): Minimum relevance score required for a document to be
            retained. Documents with scores below this threshold are discarded.
    """

    # for pedantic
    class Config:
        arbitrary_types_allowed = True

    model: CrossEncoder
    threshold: float = config.RERANKER_THRESHOLD

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """Filter retrieved documents using a Cross-Encoder relevance threshold.

        This method evaluates the semantic relevance of each document with respect
        to the input query using a Cross-Encoder model. The model assigns a relevance
        score to each (query, document) pair. Only documents with a score above the
        configured threshold are retained.

        This reranking step improves retrieval quality by removing low-relevance
        documents before passing context to the LLM, reducing noise and improving
        response accuracy in Retrieval-Augmented Generation (RAG) pipelines.

        Args:
            documents (Sequence[Document]): List of retrieved documents to evaluate.
                Each document typically represents a text chunk from the vector store.
            query (str): User query used to assess document relevance.
            callbacks (Optional[Any]): Optional LangChain callbacks for tracing or
                monitoring. Not used directly in this implementation.

        Returns:
            Sequence[Document]: Filtered list of documents whose Cross-Encoder
            relevance score exceeds the configured threshold.
        """
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        docs_with_scores = zip(documents, scores)
        result = [doc for doc, score in docs_with_scores if score > self.threshold]
        return result


@st.cache_resource
def load_vector_store() -> FAISS:
    """Load and cache the FAISS vector store with HyDE-enhanced embeddings.

    This function initializes the base embedding model and wraps it with a
    Hypothetical Document Embedder (HyDE), which uses a local LLM to generate
    synthetic documents that improve query embedding quality. The FAISS index
    is then loaded from disk using these embeddings.

    The HyDE technique enhances semantic retrieval by transforming user queries
    into richer hypothetical documents before embedding, improving recall and
    retrieval robustness, especially for complex or underspecified queries.

    Returns:
        FAISS: Loaded FAISS vector store configured with HyDE embeddings,
        ready for similarity search and retrieval.
    """
    print("Loading vector store and embedding model...")
    llm = Ollama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL_NAME,
        temperature=config.TEMPERATURE,
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": config.DEVICE},
    )

    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=llm, base_embeddings=embeddings, prompt_key="web_search"
    )
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH,
        hyde_embeddings,
        allow_dangerous_deserialization=True,
    )
    print("Loading complete.")
    return vector_store


@st.cache_resource
def create_rag_chain(_vector_store: FAISS) -> RetrievalQA:
    """Create the complete RAG pipeline with Cross-Encoder reranking.

    This function initializes the local LLM client, configures the prompt
    template, and builds a RetrievalQA chain using the provided FAISS vector
    store as the retriever. The retriever performs similarity search over
    embedded document chunks, then applies a Cross-Encoder reranker via a
    ContextualCompressionRetriever to filter out low-relevance documents
    to provide relevant context to the LLM.
    Args:
        _vector_store (FAISS): Loaded FAISS vector store used for initial
            semantic retrieval of candidate documents.

    Returns:
        RetrievalQA: Fully configured RAG chain with HyDE retrieval,
        Cross-Encoder reranking, and LLM generation.
    """
    print("Creating RAG chain with ThresholdReranker...")
    llm = Ollama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL_NAME,
        temperature=config.TEMPERATURE,
    )
    base_retriever = _vector_store.as_retriever(search_kwargs={"k": 20})

    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = ThresholdReranker(
        model=cross_encoder_model, threshold=config.RERANKER_THRESHOLD
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    prompt = PromptTemplate(
        template=config.RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    print("RAG chain created.")
    return rag_chain


# --- Streamlit App ---
st.set_page_config(page_title="University Document Q&A", layout="wide")
st.title("🎓 University Document Q&A System")

st.sidebar.title("Options")
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

vector_store = load_vector_store()
rag_chain = create_rag_chain(vector_store)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("Ask a question about the university documents..."):
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_question)
        answer = response["result"]
        source_docs = response["source_documents"]

        full_response = f"**Answer:**\n{answer}"
        st.chat_message("assistant").markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        if debug_mode:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🔍 Debug Information")

            st.sidebar.markdown(
                f"**{len(source_docs)} relevant chunk(s) passed the threshold:**"
            )

            for i, doc in enumerate(source_docs, start=1):
                source = doc.metadata.get("source", "Unknown")
                start_page = doc.metadata.get("start_page", "N/A")
                end_page = doc.metadata.get("end_page", "N/A")

                page_info = f"Page: {start_page}"
                if start_page != end_page:
                    page_info = f"Pages: {start_page}-{end_page}"

                with st.sidebar.expander(
                    f"Chunk {i} | Source: {os.path.basename(source)} ({page_info})"
                ):
                    st.text(doc.page_content)

            with st.sidebar.expander("Final Prompt to LLM"):
                final_prompt = config.RAG_PROMPT_TEMPLATE.format(
                    context="\n\n---\n\n".join(doc.page_content for doc in source_docs),
                    question=user_question,
                )
                st.text(final_prompt)
