# config.py

DEVICE = "cuda"

# --- Paths ---
PDF_DATA_PATH = "data"

PROCESSED_DATA_PATH = "marker_processed_data"
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
TEMPERATURE = 0.0

# --- RAG Prompt Template ---
# This is the prompt that will be sent to the LLM with the retrieved context
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant for university students and staff. Your task is to answer the user's question based ONLY on the provided context.
Synthesize the information from the context into a clear and coherent answer. Do not just repeat the source text.
Answer in complete, well-formed sentences. Answer in the language of the question.


CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

MULTY_RETRIEVER_PROMPT_TEMPLATE = """You are an AI assistant. Given a user question, do the following:

1. If the question contains multiple topics separated by "and" or other conjunctions, split it into separate sub-questions.
2. Keep all names, acronyms, and technical terms exactly as they appear; do not interpret, translate, or modify them.
3. Number each sub-question starting from 1.
4. Do not create more sub-questions than necessary.
5. Separate each sub-question with a newline.

Example1:
Input: "What is the weather in AZS and what are the latest news about aj?"
Output:
1. What is the weather in AZS?
2. What are the latest news about aj?
Example2:
Input: "What does the CCU do and how does it work?"
Output:
1. What does the CCU do?
2. How does CCU work?
Example3:
Input: "I want informations on AVGH. For APPO is a HS needed?"
Output:
1. I want information on AVGH.
2. For APPO, is a HS needed?

Original question: {question}

"""
