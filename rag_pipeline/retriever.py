"""
Retriever utilities for the RAG pipeline.

This module provides functions to retrieve relevant context chunks from a ChromaDB collection
given a query embedding. Used as part of the retrieval-augmented generation (RAG) workflow.
"""

import logging
from typing import List
from chromadb.api.models.Collection import Collection
from chroma_db.chroma import retrieve_similar_items  

logger = logging.getLogger(__name__)

def retrieve_context(collection: Collection, query_embedding: List[float], k: int = 3) -> List[str]:
    """
    Retrieves top-k relevant context chunks from Chroma given a query embedding.

    Parameters
    ----------
    collection : Collection
        The Chroma collection object to query from.
    query_embedding : List[float]
        The vector representation of the query.
    k : int
        The number of top similar items to retrieve.

    Returns
    -------
    List[str]
        A list of document strings (context chunks).
    """
    try:
        # Use existing robust Chroma wrapper
        similar_items = retrieve_similar_items(collection, query_embedding, n_results=k)

        if not similar_items:
            logger.warning("No similar items retrieved for the given query.")
            return []

        # Extract documents
        context_chunks = [item["document"] for item in similar_items]

        # Count how many of the retrieved items were chunks
        chunk_count = sum(1 for item in similar_items if item["metadata"].get("chunk_index") is not None)
        logger.info("Retrieved %d context chunks from Chroma.", len(context_chunks))
        
        # Log how many out of the retrieved items were chunks
        logger.info("Out of %d retrieved items, %d were valid chunks.", len(similar_items), chunk_count)
        for i, item in enumerate(similar_items):
            logger.debug(f"Chunk {i+1}: ID={item['id']} | Distance={item['distance']:.4f}")

        return context_chunks

    except Exception as e:
        logger.error(f"Error in retrieve_context: {e}")
        return []