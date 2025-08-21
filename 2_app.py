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
        model_kwargs={"device": config.DEVICE},  # Use config.DEVICE
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
    llm = Ollama(
        base_url=config.OLLAMA_BASE_URL,
        model=config.OLLAMA_MODEL_NAME,
        temperature=config.TEMPERATURE,
    )

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

# --- NEW: Add a debug mode toggle in the sidebar ---
st.sidebar.title("Options")
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)


# Load resources
vector_store = load_vector_store()
rag_chain = create_rag_chain(vector_store)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])  # Allow HTML for expander

# React to user input
if user_question := st.chat_input("Ask a question about the university documents..."):
    # Display user message in chat message container
    st.chat_message("user").text(user_question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Thinking..."):
        # Get the answer from the RAG chain
        response = rag_chain.invoke(user_question)
        answer = response["result"]
        source_docs = response["source_documents"]

        # Format the main response with sources
        full_response = f"**Answer:**\n{answer}\n\n"
        full_response += "**Sources:**\n"
        unique_sources = set(
            doc.metadata.get("source", "Unknown") for doc in source_docs
        )
        for i, source in enumerate(unique_sources):
            full_response += f"- {source}\n"

        if debug_mode:
            # Build a numbered list of chunks with clear dividers
            chunk_markdown_list = []
            for i, doc in enumerate(source_docs, start=1):
                source = doc.metadata.get("source", "Unknown")
                chunk_markdown_list.append(
                    f"**Chunk {i} — Source:** `{source}`\n\n```\n{doc.page_content}\n```"
                )
            context_text_markdown = "\n\n---\n\n".join(chunk_markdown_list)

            # Reconstruct the final prompt for display
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

            # Show debug info in its own assistant message
            with st.chat_message("assistant"):
                st.markdown(debug_markdown)

            st.session_state.messages.append(
                {"role": "assistant", "content": debug_markdown}
            )

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.text(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
