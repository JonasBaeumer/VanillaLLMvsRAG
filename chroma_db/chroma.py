from pathlib import Path
import logging
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,            
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def initiate_chroma_db(directory_path: str) -> chromadb.api.client:
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
    Store a single item in the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to store the item in.
    item_id : str
        Unique identifier for the item.
    item_data : dict
        Data to be stored in the item.
    """
    try:
        collection.add(
            ids=[item_data.get("id", "")],
            documents=[item_data.get("document", "")],
            metadatas=[item_data.get("metadata", {})],
            embeddings=[item_data.get("embedding", [])]
        )
        logger.info(f"Item {item_id} stored successfully.")
    except Exception as e:
        logger.error(f"Error storing item {item_id}: {e}")


def store_multiple_items(collection, items: list):
    """
    Store multiple items in the ChromaDB collection.

    Parameters
    ----------
    collection : chromadb.api.Collection
        The ChromaDB collection to store the items in.
    items : list
        List of dictionaries containing item data.
    """
    try:
        ids = [item.get("id") for item in items]
        documents = [item.get("document", "") for item in items]
        metadatas = [item.get("metadata", {}) for item in items]
        embeddings = [item.get("embedding", []) for item in items]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        logger.info(f"{len(items)} items stored successfully.")
    except Exception as e:
        logger.error(f"Error storing multiple items: {e}")


def retrieve_item(collection, item_id: str):
    """
    Retrieve a single item from the ChromaDB collection.

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
    

def retrieve_multiple_items(collection, item_ids: list):
    """
    Retrieve multiple items from the ChromaDB collection.

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
    list
        List of similar item data.
    """
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results['documents'] if results and results['documents'] else []
    except Exception as e:
        logger.error(f"Error retrieving similar items: {e}")
        return []


if __name__ == "__main__":
    
    # UNCOMMENT ONLY ON THE FIRST RUN TO INITIALIZE VECTOR STORAGE
    # initiate_chroma_db("./chroma_db")
    # print("ChromaDB initialized successfully.")




