from typing import Any
from uuid import uuid4

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun


class MultiQueryRerankRetriever(MultiQueryRetriever):
    reranker: Any

    def __init__(self, base_retriever, llm, reranker: Any):
        super().__init__(retriever=base_retriever, llm_chain=llm, reranker=reranker)

    def get_relevant_documents_multi_query(self, query, run_manager=None):
        if run_manager is None:
            run_manager = CallbackManagerForRetrieverRun(
                run_id=uuid4(),
                handlers=[],  # aucun handler custom
                inheritable_handlers=[],  # pas d’héritage
            )
        subqueries = self.generate_queries(query, run_manager)
        all_docs = []
        all_scores = []
        for subquery in subqueries:
            docs = self.retriever.invoke(subquery)
            reranked_docs = self.reranker.rerank(docs, subquery)
            all_docs.append(reranked_docs["docs"])
            all_scores.append(reranked_docs["scores"])
        return all_docs, subqueries, all_scores
