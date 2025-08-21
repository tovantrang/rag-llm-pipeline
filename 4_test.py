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
# --- Imports LangChain ---
# --- Import du modèle de Re-Ranking ---


# #####################################################################################
# 1. DÉFINITION DE NOTRE RE-RANKER PERSONNALISÉ AVEC SEUIL
# #####################################################################################
class ThresholdReranker:
    def __init__(self, model, threshold=0.3, k=5):
        self.model = model
        self.threshold = threshold
        self.k = k

    def rerank(self, docs, query):
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        docs_with_scores = zip(docs, scores)
        results = {"docs": [], "scores": []}
        for doc, score in docs_with_scores:
            sigmoid = 1 / (1 + math.exp(-score))  # Apply sigmoid to score
            if sigmoid > self.threshold:
                results["docs"].append(doc)
                results["scores"].append((score, sigmoid))
        return results


class CustomRetrievalQA:
    def __init__(self, llm, chain_type, retriever, prompt):
        self.llm = llm
        self.chain_type = chain_type
        self.retriever = retriever
        self.prompt = prompt

    def invoke(self, query):
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


# #####################################################################################
# 2. FONCTIONS DE CHARGEMENT ET DE CRÉATION DE LA CHAÎNE RAG
# #####################################################################################


@st.cache_resource
def load_vector_store():
    """Charge la base vectorielle FAISS depuis le disque."""
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
    """
    Crée la chaîne RAG complète en utilisant le retriever de base et notre
    re-ranker personnalisé.
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
    reranker = ThresholdReranker(model=cross_encoder_model, threshold=0.5, k=5)
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


# #####################################################################################
# 3. INTERFACE UTILISATEUR STREAMLIT
# #####################################################################################

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
