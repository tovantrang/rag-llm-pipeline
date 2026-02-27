# app.py
import math
import os

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import LineListOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from sentence_transformers.cross_encoder import CrossEncoder

import config
from multiretriever import MultiQueryRerankRetriever


class ThresholdReranker:
    """Rerank and filter retrieved documents using a Cross-Encoder score threshold.

    This component scores each (query, document) pair using a Cross-Encoder model,
    applies a sigmoid transformation to obtain values in [0, 1], and retains only
    documents whose sigmoid score exceeds the configured threshold.

    It returns both the filtered documents and their associated raw and sigmoid
    scores for debugging and inspection.

    Attributes:
        model: Cross-Encoder model exposing a `predict(pairs)` method returning
            relevance logits/scores for (query, text) pairs.
        threshold (float): Minimum sigmoid score required to keep a document.
        k (int): Intended maximum number of documents to keep. Note: in the
            current implementation, `k` is stored but not enforced.
    """

    def __init__(self, model, threshold=0.3, k=5):
        self.model = model
        self.threshold = threshold
        self.k = k

    def rerank(self, docs, query):
        """Score, filter, and return documents relevant to the query.

        Args:
            docs: Candidate documents (typically retrieved from a vector store).
                Each document must expose a `page_content` string attribute.
            query (str): Query text used to compute relevance scores.

        Returns:
            dict: A dictionary with:
                - "docs": Filtered documents whose sigmoid(score) > threshold.
                - "scores": List of tuples (raw_score, sigmoid_score) aligned
                  with "docs".
        """
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        docs_with_scores = zip(docs, scores)
        results = {"docs": [], "scores": []}
        for doc, score in docs_with_scores:
            sigmoid = 1 / (1 + math.exp(-score))
            if sigmoid > self.threshold:
                results["docs"].append(doc)
                results["scores"].append((score, sigmoid))
        return results


class CustomRetrievalQA:
    """Custom Retrieval-Augmented Generation (RAG) chain with multi-query support.

    This class orchestrates a RAG workflow by:
      1. Expanding the original query into multiple subqueries (handled by the
         injected retriever).
      2. Retrieving documents for each subquery.
      3. Building a context string from retrieved document chunks.
      4. Formatting a prompt with (context, question) and calling the LLM.
      5. Returning rich debug outputs (subqueries, prompts, scores, sources).

    This is designed for transparency and debugging: it surfaces intermediate
    artifacts that are typically hidden in higher-level LangChain chains.

    Attributes:
        llm: Callable LLM interface. Must be callable as `llm(prompt_text)`.
        chain_type (str): Chain composition strategy indicator (e.g., "stuff").
            Stored for bookkeeping; not used directly in the current implementation.
        retriever: Retriever object implementing
            `get_relevant_documents_multi_query(query)`.
        prompt: Optional prompt template exposing `format(context=..., question=...)`.
    """

    def __init__(self, llm, chain_type, retriever, prompt):
        self.llm = llm
        self.chain_type = chain_type
        self.retriever = retriever
        self.prompt = prompt

    def invoke(self, query):
        """Run the RAG pipeline for a user query and return detailed results.

        Args:
            query (str): User query.

        Returns:
            dict: Dictionary containing:
                - "query": Original user query.
                - "responses": LLM responses, one per generated subquery.
                - "source_documents": Retrieved documents per subquery.
                - "subqueries": Generated subqueries.
                - "prompts": Final prompts sent to the LLM (for debugging).
                - "scores": Reranker scores per subquery (format depends on retriever).
        """
        results = {
            "query": query,
            "responses": [],
            "source_documents": [],
            "subqueries": [],
            "prompts": [],
            "scores": [],
        }
        all_docs, subqueries, all_scores = (
            self.retriever.get_relevant_documents_multi_query(query)
        )
        for docs, subquery, scores in zip(all_docs, subqueries, all_scores):
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)
            if self.prompt:
                prompt_text = self.prompt.format(context=context, question=subquery)
            else:
                prompt_text = subquery
            print(f"Prompt: {prompt_text}")
            response = self.llm(prompt_text)
            results["source_documents"].append(docs)
            results["responses"].append(response)
            results["subqueries"].append(subquery)
            results["prompts"].append(prompt_text)
            results["scores"].append(scores)
        return results


@st.cache_resource
def load_vector_store():
    """Load and cache the FAISS vector store from disk.

    This function initializes the embedding model used by the vector store and
    loads a FAISS index from local storage. The embedding configuration must be
    consistent with the one used at ingestion time to ensure correct similarity
    search behavior.

    The result is cached using Streamlit's resource cache to avoid repeated disk
    I/O and model initialization across app reruns.

    Returns:
        FAISS: Loaded FAISS vector store ready for retrieval.
    """
    print("Loading vector store and embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": config.DEVICE},
    )
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("Loading complete.")
    return vector_store


@st.cache_resource
def create_rag_chain(_vector_store):
    """Create and cache the full RAG chain with multi-query retrieval and reranking.

    This function builds a complete RAG pipeline:
      - Base retrieval from the FAISS vector store (top-k candidates).
      - Multi-query generation using an LLM chain (prompt → LLM → line parser).
      - Cross-Encoder reranking and filtering using `ThresholdReranker`.
      - Final answer generation using a context-injected prompt ("stuff" strategy).

    The resulting chain is cached using Streamlit's resource cache to prevent
    redundant initialization across app reruns.

    Args:
        _vector_store (FAISS): Loaded FAISS vector store used to create the base retriever.

    Returns:
        CustomRetrievalQA: Configured RAG chain that returns detailed intermediate
        artifacts (subqueries, prompts, scores, source documents).
    """
    print("Creating RAG chain with ThresholdReranker...")
    llm = Ollama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL_NAME,
        temperature=config.TEMPERATURE,
    )

    base_retriever = _vector_store.as_retriever(search_kwargs={"k": 20})

    retrival_prompt = PromptTemplate(
        template=config.MULTY_RETRIEVER_PROMPT_TEMPLATE,
        input_variables=["question"],
    )
    output_parser = LineListOutputParser()
    llm_chain = retrival_prompt | llm | output_parser
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = ThresholdReranker(
        model=cross_encoder_model, threshold=config.RERANKER_THRESHOLD, k=5
    )
    retriever = MultiQueryRerankRetriever(base_retriever, llm_chain, reranker)

    prompt = PromptTemplate(
        template=config.RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    rag_chain = CustomRetrievalQA(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        prompt=prompt,
    )
    print("RAG chain created.")
    return rag_chain


# --- Streamlit App ---
st.set_page_config(page_title="University Document Q&A", layout="wide")
st.title("🎓 University Document Q&A System")

st.sidebar.title("Options")
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)
if debug_mode:
    show_prompts = st.sidebar.checkbox("Show prompts", value=False)

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
        results = rag_chain.invoke(user_question)
        answers_list = results["responses"]
        source_docs_list = results["source_documents"]
        subqueries_list = results["subqueries"]
        prompts_list = results["prompts"]
        scores_list = results["scores"]

        full_response = "\n\n".join(
            f"**Answer for subquery '{subquery}':**\n{answer}"
            for subquery, answer in zip(subqueries_list, answers_list)
        )
        st.chat_message("assistant").markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

        if debug_mode:
            st.header("🔍 Debug Information")

            for sub_i, (source_docs, subquery, prompt, scores) in enumerate(
                zip(source_docs_list, subqueries_list, prompts_list, scores_list),
                1,
            ):
                st.subheader(
                    f"Subquery {sub_i}: {subquery} | Documents: {len(source_docs)}"
                )
                for chunk_i, (doc, (score, sigmoid)) in enumerate(
                    zip(source_docs, scores), 1
                ):
                    source = doc.metadata.get("source", "Unknown")
                    start_page = doc.metadata.get("start_page", "N/A")
                    end_page = doc.metadata.get("end_page", "N/A")
                    page_info = (
                        f"Page: {start_page}"
                        if start_page == end_page
                        else f"Pages: {start_page}-{end_page}"
                    )

                    with st.expander(
                        f"Chunk {chunk_i} | Score: {score:.2f} | Sigmoid: {sigmoid:.3f} | Source: {os.path.basename(source)} ({page_info})"
                    ):
                        st.text(doc.page_content)
                if show_prompts:
                    with st.expander("**Prompt:**"):
                        st.markdown(prompt)
