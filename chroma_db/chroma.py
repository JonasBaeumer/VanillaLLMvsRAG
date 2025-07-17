"""
ChromaDB utility functions for vector storage and retrieval.

This module provides helper functions to initialize, store, retrieve, update, and delete items in a ChromaDB collection.
It is used for managing vector-based document storage and similarity search in the RAG pipeline.
"""
from pathlib import Path
import logging
import chromadb
from chromadb.config import Settings
import time

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,            
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def initiate_chroma_db(directory_path: str) -> chromadb.PersistentClient:
    """
    Initialise a persistent ChromaDB client with basic error handling.

    Parameters
    ----------
    db_dir : str
        Directory where Chroma should store its DuckDB + Parquet files.
    fallback_to_memory : bool
        If True, returns an in-memory client on failure instead of raising.

    Returns
    -------
    chromadb.api.Client
        Ready-to-use Chroma client (persistent or in-memory).
    """

    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logger.exception("Could not create Chroma directory %s", directory_path)

    try:
        client = chromadb.PersistentClient(      
            path=directory_path,                         
            settings=Settings()                  
        )
        # create (or load) a vector collection
        papers = client.get_or_create_collection("papers")
        logger.info("Chroma initialised with persistent store at %s", directory_path)

    except Exception as err:
        logger.exception("Failed to initialise persistent Chroma at %s", directory_path)
    
    return client


def store_single_item(collection, item_data: dict):
    """
    Store a single item in the ChromaDB collection, with duplicate check.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to store the item in.
    item_data : dict
        Data to be stored in the item.
    """
    item_id = item_data.get("id", "")
    try:
        if item_exists(collection, item_id):
            logger.warning(f"Item {item_id} already exists — skipping insert.")
            return

        collection.add(
            ids=[item_id],
            documents=[item_data.get("document", "")],
            metadatas=[item_data.get("metadata", {})],
            embeddings=[item_data.get("embedding", [])]
        )
        logger.info(f"Item {item_id} stored successfully.")

    except Exception as e:
        logger.error(f"Error storing item {item_id}: {e}")


def store_multiple_items(collection, items: list):
    """
    Store multiple items in the ChromaDB collection, with duplicate check.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to store the items in.
    items : list
        List of dictionaries containing item data.
    """
    try:
        # Filter out existing items
        new_items = []
        for item in items:
            item_id = item.get("id")
            if not item_exists(collection, item_id):
                new_items.append(item)
            else:
                logger.warning(f"Item {item_id} already exists — skipping insert.")

        if not new_items:
            logger.info("No new items to store (all were duplicates).")
            return

        # Prepare batch insert for new items
        ids = [item.get("id") for item in new_items]
        documents = [item.get("document", "") for item in new_items]
        metadatas = [item.get("metadata", {}) for item in new_items]
        embeddings = [item.get("embedding", []) for item in new_items]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"{len(new_items)} new items stored successfully.")

    except Exception as e:
        logger.error(f"Error storing multiple items: {e}")


def get_item(collection, item_id: str):
    """
    Get a single item from the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to retrieve the item from.
    item_id : str
        Unique identifier for the item to retrieve.

    Returns
    -------
    dict
        The retrieved item data.
    """
    try:
        results = collection.get(ids=[item_id])
        if results and results['documents']:
            return results['documents'][0]
        else:
            logger.warning(f"Item {item_id} not found.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving item {item_id}: {e}")
        return None
    

def get_multiple_items(collection, item_ids: list):
    """
    Get multiple items from the ChromaDB collection by item id.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to retrieve the items from.
    item_ids : list
        List of unique identifiers for the items to retrieve.

    Returns
    -------
    list
        List of retrieved item data.
    """
    try:
        results = collection.get(ids=item_ids)
        return results['documents'] if results and results['documents'] else []
    except Exception as e:
        logger.error(f"Error retrieving multiple items: {e}")
        return []
    

def retrieve_similar_items(collection, query_embedding: list, n_results: int = 5):
    """
    Retrieve similar items based on a query embedding.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to search in.
    query_embedding : list
        The embedding to search for similar items.
    n_results : int
        Number of similar items to retrieve.

    Returns
    -------
    list of dict
        List of similar item data:
        [{"id": ..., "document": ..., "metadata": ..., "distance": ...}, ...]
    """
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if not results or not results['documents']:
            return []

        # Unpack results
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # Build structured result list
        return [
            {
                "id": id_,
                "document": doc,
                "metadata": meta,
                "distance": dist
            }
            for id_, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]

    except Exception as e:
        logger.error(f"Error retrieving similar items: {e}")
        return []


def delete_item(collection, item_id: str):
    """
    Delete a single item from the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to delete the item from.
    item_id : str
        Unique identifier for the item to delete.
    """
    try:
        collection.delete(ids=[item_id])
        logger.info(f"Item {item_id} deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting item {item_id}: {e}")


def delete_multiple_items(collection, item_ids: list):
    """
    Delete multiple items from the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to delete the items from.
    item_ids : list
        List of unique identifiers for the items to delete.
    """
    try:
        collection.delete(ids=item_ids)
        logger.info(f"{len(item_ids)} items deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting multiple items: {e}")


def item_exists(collection, item_id: str) -> bool:
    """
    Check if an item exists in the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to check.
    item_id : str
        Unique identifier for the item to check.

    Returns
    -------
    bool
        True if the item exists, False otherwise.
    """
    try:
        results = collection.get(ids=[item_id])
        # Check if first list of documents contains anything
        exists = bool(results and results['documents'] and results['documents'][0])
        return exists
    except Exception as e:
        logger.error(f"Error checking existence of item {item_id}: {e}")
        return False
    

def list_all_ids(collection) -> list:
    """
    List all item IDs in the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to list IDs from.

    Returns
    -------
    list
        List of all item IDs in the collection.
    """
    try:
        results = collection.get()
        ids = results['ids'] if results and 'ids' in results else []
        logger.info(f"Retrieved {len(ids)} item IDs.")
        return ids
    except Exception as e:
        logger.error(f"Error listing all IDs: {e}")
        return []
    

def update_item(collection, item_id: str, item_data: dict):
    """
    Update an existing item in the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to update the item in.
    item_id : str
        Unique identifier for the item to update.
    item_data : dict
        Data to update in the item.
    """
    if not item_exists(collection, item_id):
        logger.error(f"Item {item_id} does not exist. Cannot update.")
        return

    try:
        collection.update(
            ids=[item_id],
            documents=[item_data.get("document", "")],
            metadatas=[item_data.get("metadata", {})],
            embeddings=[item_data.get("embedding", [])]
        )
        logger.info(f"Item {item_id} updated successfully.")
    except Exception as e:
        logger.error(f"Error updating item {item_id}: {e}")


def clear_collection(collection):
    """
    Clear all items from the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to clear.
    """
    result = collection.get()
    all_ids = result["ids"][0]

    if not all_ids:
        logger.info("Collection is already empty.")
        return
    
    try:
        collection.delete(ids=all_ids)
        unique_ids = set(all_ids)
        logger.info(f"Deleted {len(unique_ids)} unique items ({len(all_ids)} total embeddings) from collection '{collection.name}'.")

    except Exception as e:
        logger.error(f"Error clearing collection: {e}")
        

#if __name__ == "__main__":
    # LOCAL TESTING ONLY: The following block is for manual/local testing and should not be run in production or on import.
    # from sentence_transformers import SentenceTransformer
    # # 1️⃣ Load embedding model (same for indexing & query)
    # embed_model = SentenceTransformer("BAAI/bge-large-en")
    # # 2️⃣ Initiate Chroma DB and collection
    # db_dir = "./chroma_db"
    # client = initiate_chroma_db(db_dir)
    # collection = client.get_or_create_collection("papers")
    # # 3️⃣ Define mock paper data
    # papers = [
    #     {
    #         "id": "paper-001",
    #         "document": "Transformers are neural networks designed to process sequential data using attention mechanisms.",
    #         "metadata": {"title": "Attention is All You Need", "year": 2017},
    #         "embedding": embed_model.encode("Transformers are neural networks designed to process sequential data using attention mechanisms.", normalize_embeddings=True).tolist()
    #     },
    #     {
    #         "id": "paper-002",
    #         "document": "Retrieval-augmented generation improves language models by grounding them on external documents.",
    #         "metadata": {"title": "RAG: Retrieval-Augmented Generation", "year": 2020},
    #         "embedding": embed_model.encode("Retrieval-augmented generation improves language models by grounding them on external documents.", normalize_embeddings=True).tolist()
    #     },
    #     {
    #         "id": "paper-003",
    #         "document": "Vector databases enable fast similarity search in high-dimensional embedding spaces.",
    #         "metadata": {"title": "Vector Search at Scale", "year": 2022},
    #         "embedding": embed_model.encode("Vector databases enable fast similarity search in high-dimensional embedding spaces.", normalize_embeddings=True).tolist()
    #     }
    # ]
    # # 4️⃣ Store papers (first insert → should store them)
    # print("\n=== First insert of papers ===")
    # collection.add(
    #     documents=[p["document"] for p in papers],
    #     metadatas=[p["metadata"] for p in papers],
    #     ids=[p["id"] for p in papers],
    #     embeddings=[p["embedding"] for p in papers],
    # )