# config.py

DEVICE = "cuda"

# --- Paths ---
PDF_DATA_PATH = "data"
VECTOR_STORE_PATH = "vector_store"

# --- Text Processing & Chunking ---
# Using a model that supports English and Traditional Chinese
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128

# --- Vector Store ---
# Number of relevant chunks to retrieve for context
K_RETRIEVED_CHUNKS = 5

# --- Local LLM ---
# The model name as served by Ollama
OLLAMA_MODEL_NAME = "mistral"
# The base URL for the Ollama server
OLLAMA_BASE_URL = "http://localhost:11434"

# --- RAG Prompt Template ---
# This is the prompt that will be sent to the LLM with the retrieved context
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for university students and staff. Your task is to answer the user's question based ONLY on the provided context.
If the context does not contain the answer, state that the information is not available in the provided documents. Do not use any external knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
