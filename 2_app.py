# app.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

import config


@st.cache_resource
def load_vector_store() -> FAISS:
    """Load the FAISS vector store with its embedding model.

    This function initializes the embedding model and loads the FAISS index
    from local storage. The embedding model must match the one used during
    ingestion to ensure consistent vector representations and correct
    similarity search behavior.

    Returns:
        FAISS: Loaded FAISS vector store ready for similarity search.
    """
    print("Loading vector store and embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": config.DEVICE},  # Use config.DEVICE
    )
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # Required for FAISS with LangChain
    )
    print("Loading complete.")
    return vector_store


@st.cache_resource
def create_rag_chain(vector_store: FAISS) -> RetrievalQA:
    """Create the RetrievalQA chain for the RAG pipeline.

    This function initializes the local LLM client, configures the prompt
    template, and builds a RetrievalQA chain using the provided FAISS vector
    store as the retriever. The retriever performs similarity search over
    embedded document chunks to provide relevant context to the LLM.

    Args:
        vector_store (FAISS): Loaded FAISS vector store used to retrieve
            relevant document chunks based on semantic similarity.

    Returns:
        RetrievalQA: Configured RetrievalQA chain ready to answer user queries,
        including retrieved source documents.
    """
    print("Creating RAG chain...")
    llm = Ollama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL_NAME,
        temperature=config.TEMPERATURE,
    )

    prompt = PromptTemplate(
        template=config.RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # context directly into the prompt
        retriever=vector_store.as_retriever(
            search_kwargs={"k": config.K_RETRIEVED_CHUNKS}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    print("RAG chain created.")
    return rag_chain


# --- Streamlit App ---
st.set_page_config(page_title="University Document Q&A", layout="wide")
st.title("🎓 University Document Q&A System")

st.sidebar.title("Options")
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

vector_store = load_vector_store()
rag_chain = create_rag_chain(vector_store)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

if user_question := st.chat_input("Ask a question about the university documents..."):
    st.chat_message("user").text(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Thinking..."):
        response = rag_chain.invoke(user_question)
        answer = response["result"]
        source_docs = response["source_documents"]

        full_response = f"**Answer:**\n{answer}\n\n"
        full_response += "**Sources:**\n"
        unique_sources = set(
            doc.metadata.get("source", "Unknown") for doc in source_docs
        )
        for i, source in enumerate(unique_sources):
            full_response += f"- {source}\n"

        if debug_mode:
            chunk_markdown_list = []
            for i, doc in enumerate(source_docs, start=1):
                source = doc.metadata.get("source", "Unknown")
                chunk_markdown_list.append(
                    f"**Chunk {i} — Source:** `{source}`\n\n```\n{doc.page_content}\n```"
                )
            context_text_markdown = "\n\n---\n\n".join(chunk_markdown_list)

            final_prompt = config.RAG_PROMPT_TEMPLATE.format(
                context="\n\n".join(doc.page_content for doc in source_docs),
                question=user_question,
            )

            debug_markdown = (
                "### 🔍 Debug Mode\n\n"
                "**Retrieved Context Chunks:**\n\n"
                f"{context_text_markdown}\n\n"
                "---\n\n"
                "**Final Prompt to LLM:**\n\n"
                f"```\n{final_prompt}\n```"
            )

            with st.chat_message("assistant"):
                st.markdown(debug_markdown)

            st.session_state.messages.append(
                {"role": "assistant", "content": debug_markdown}
            )

        with st.chat_message("assistant"):
            st.text(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
