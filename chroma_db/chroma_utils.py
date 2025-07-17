"""
ChromaDB utility functions for loading collections.

This module provides helper functions to load or create persistent ChromaDB collections for vector storage.
"""
import chromadb
from chromadb.config import Settings

def load_collection(collection_name: str, db_path: str = "./chroma_db"):
    """
    Loads (or creates if not existing) a persistent Chroma collection.

    Parameters
    ----------
    collection_name : str
        Name of the collection to load.
    db_path : str
        Path where the Chroma DB files are stored.

    Returns
    -------
    chromadb.api.models.Collection.Collection
        The Chroma collection object.
    """
    # Initialize persistent Chroma client
    client = chromadb.PersistentClient(path=db_path, settings=Settings())
    collection = client.get_or_create_collection(collection_name)

    return collection