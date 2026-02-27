from typing import Any
from uuid import uuid4

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class MultiQueryRerankRetriever(MultiQueryRetriever):
    """Multi-query retriever with Cross-Encoder reranking and score filtering.

    This retriever extends LangChain's MultiQueryRetriever by:
      1) Generating multiple subqueries from an initial user query using an LLM chain.
      2) Retrieving candidate documents for each subquery from a base retriever
         (e.g., a FAISS retriever).
      3) Applying a reranker (typically a Cross-Encoder) to score and filter the
         retrieved documents per subquery.

    The method `get_relevant_documents_multi_query` returns per-subquery results
    (documents and scores) to support downstream pipelines that need full
    transparency (e.g., debugging prompts, attribution, or analysis).

    Attributes:
        reranker (Any): Reranker component exposing a `rerank(docs, query)` method
            returning a dict with keys "docs" (filtered documents) and "scores"
            (associated scoring metadata).
    """

    reranker: Any

    def __init__(self, base_retriever, llm, reranker: Any):
        super().__init__(retriever=base_retriever, llm_chain=llm, reranker=reranker)

    def get_relevant_documents_multi_query(
        self, query, run_manager=None
    ) -> tuple[list[list[Document]], list[str], list[Any]]:
        """Retrieve documents for multiple generated subqueries and rerank results.

        This method generates subqueries from the input query, retrieves documents
        for each subquery using the underlying base retriever, and then applies the
        configured reranker to filter and reorder documents by relevance.

        If no `run_manager` is provided, a default `CallbackManagerForRetrieverRun`
        instance is created to satisfy the MultiQueryRetriever interface.

        Args:
            query (str): User query used to generate subqueries and drive retrieval.
            run_manager (Optional[CallbackManagerForRetrieverRun]): Optional LangChain
                callback manager for tracing and instrumentation.

        Returns:
            tuple[list[list[Document]], list[str], list[Any]]: A tuple containing:
                - all_docs: List of document lists, one list per generated subquery,
                after reranking/filtering.
                - subqueries: List of generated subqueries.
                - all_scores: List of reranker score outputs aligned with `all_docs`.
                The exact score format depends on the reranker implementation.
        """
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
