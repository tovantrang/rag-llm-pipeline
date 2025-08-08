# app.py
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

import config


# Function to load the vector store
@st.cache_resource
def load_vector_store():
    """Loads the FAISS vector store and the embedding model."""
    print("Loading vector store and embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # Use 'cuda' if GPU is available
    )
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # Required for FAISS with LangChain
    )
    print("Loading complete.")
    return vector_store


# Function to create the RAG chain
def create_rag_chain(vector_store):
    """Creates the RetrievalQA chain."""
    # Initialize the local LLM
    llm = Ollama(base_url=config.OLLAMA_BASE_URL, model=config.OLLAMA_MODEL_NAME)

    # Create the prompt from the template
    prompt = PromptTemplate(
        template=config.RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # Create the RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": config.K_RETRIEVED_CHUNKS}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return rag_chain


# --- Streamlit App ---
st.set_page_config(page_title="University Document Q&A", layout="wide")
st.title("🎓 University Document Q&A System")

# Load resources
vector_store = load_vector_store()
rag_chain = create_rag_chain(vector_store)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the university documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Get the answer from the RAG chain
        response = rag_chain.invoke(prompt)
        answer = response["result"]
        source_docs = response["source_documents"]

        # Format the response with sources
        full_response = f"**Answer:**\n{answer}\n\n"
        full_response += "**Sources:**\n"
        # Create a unique list of sources
        unique_sources = set(
            doc.metadata.get("source", "Unknown") for doc in source_docs
        )
        for i, source in enumerate(unique_sources):
            full_response += f"- {source}\n"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
