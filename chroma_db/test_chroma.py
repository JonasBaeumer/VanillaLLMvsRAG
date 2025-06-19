import logging
import time
import chroma_db.chroma as chroma
from models.openai_models import OpenAIEmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    
    # UNCOMMENT ONLY ON THE FIRST RUN TO INITIALIZE VECTOR STORAGE
    # initiate_chroma_db("./chroma_db")
    # print("ChromaDB initialized successfully.")

    from sentence_transformers import SentenceTransformer

    # 1️⃣ Load embedding model (same for indexing & query)
    embedder = OpenAIEmbeddingModel()

    # 2️⃣ Initiate Chroma DB and collection
    db_dir = "./chroma_db"
    client = chroma.initiate_chroma_db(db_dir)

    # ONYL UNCOMMENT TO DELETE EXISTING COLLECTION
    # client.delete_collection("papers")
    
    collection = client.get_or_create_collection("papers")

    # 3️⃣ Define mock paper data
    papers = [
        {
            "id": "paper-001",
            "document": "Transformers are neural networks designed to process sequential data using attention mechanisms.",
            "metadata": {"title": "Attention is All You Need", "year": 2017},
            "embedding": embedder.embed_text("Transformers are neural networks designed to process sequential data using attention mechanisms.")
        },
        {
            "id": "paper-002",
            "document": "Retrieval-augmented generation improves language models by grounding them on external documents.",
            "metadata": {"title": "RAG: Retrieval-Augmented Generation", "year": 2020},
            "embedding": embedder.embed_text("Retrieval-augmented generation improves language models by grounding them on external documents.")
        },
        {
            "id": "paper-003",
            "document": "Vector databases enable fast similarity search in high-dimensional embedding spaces.",
            "metadata": {"title": "Vector Search at Scale", "year": 2022},
            "embedding": embedder.embed_text("Vector databases enable fast similarity search in high-dimensional embedding spaces.")
        }
    ]

    # 4️⃣ Store papers (first insert → should store them)
    print("\n=== First insert of papers ===")
    chroma.store_multiple_items(collection, papers)
    time.sleep(0.5)

    # 5️⃣ Attempt to store same papers again (should skip all)
    print("\n=== Attempting to insert duplicate papers ===")
    chroma.store_multiple_items(collection, papers)
    time.sleep(0.5)

    # 6️⃣ Insert one duplicate + one new item
    print("\n=== Attempting to insert mixed batch (1 duplicate, 1 new) ===")
    new_paper = {
        "id": "paper-004",
        "document": "Embedding models transform text into numerical representations for downstream tasks.",
        "metadata": {"title": "Introduction to Embedding Models", "year": 2024},
        "embedding": embedder.embed_text("Embedding models transform text into numerical representations for downstream tasks.")
    }
    mixed_batch = [papers[0], new_paper]   # paper-001 duplicate, paper-004 new

    chroma.store_multiple_items(collection, mixed_batch)
    time.sleep(0.5)

    # 7️⃣ List all stored IDs
    print("\n=== Stored IDs after duplicate tests ===")
    all_ids = chroma.list_all_ids(collection)
    print(all_ids)

    # 8️⃣ Retrieve similar items
    print("\n=== Retrieving similar items for query ===")
    query_text = "How can retrieval help improve language models?"
    query_embedding = embedder.embed_text(query_text)

    similar_items = chroma.retrieve_similar_items(collection, query_embedding, n_results=3)
    for idx, item in enumerate(similar_items, 1):
        print(f"\nResult {idx}:")
        print(f"ID: {item['id']}")
        print(f"Title: {item['metadata'].get('title')}")
        print(f"Distance: {item['distance']:.4f}")

    # 9️⃣ Get single paper by ID
    print("\n=== Get single item by ID ===")
    item = chroma.get_item(collection, "paper-001")
    print(item)

    # 🔟 Delete one item
    print("\n=== Deleting one item ===")
    chroma.delete_item(collection, "paper-003")
    print("Remaining IDs:", chroma.list_all_ids(collection))

    # 🔄 Clear entire collection
    print("\n=== Clearing collection ===")
    chroma.clear_collection(collection)
    
    print("Final IDs:", chroma.list_all_ids(collection))

    print("\n=== Test run complete. ===")
    