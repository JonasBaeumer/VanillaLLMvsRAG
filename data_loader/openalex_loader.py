import requests
import time
from urllib.parse import quote
import logging
from typing import List, Dict, Any
from data_loader.constants import OPENALEX_WORKS_ENDPOINT
from models.openai_models import OpenAIEmbeddingModel
from chroma_db.chroma import store_multiple_items, initiate_chroma_db
import logging

logger = logging.getLogger("openalex-loader")
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def fetch_openalex_abstract(title, delay=1.0):
    """
    Queries OpenAlex with a paper title and returns metadata including the abstract.
    """
    query_url = f"{OPENALEX_WORKS_ENDPOINT}?search={quote(title)}"
    response = requests.get(query_url)
    time.sleep(delay)

    if response.status_code != 200:
        print(f"⚠️ Failed to fetch for: {title} — Status {response.status_code}")
        return None

    results = response.json().get("results", [])
    if not results:
        print(f"🔍 No results found for: {title}")
        return None

    best_match = results[0]
    return {
        "title": best_match.get("title"),
        "doi": best_match.get("doi"),
        "abstract": best_match.get("abstract_inverted_index"),
        "id": best_match.get("id"),
        "publication_year": best_match.get("publication_year"),
        "authorships": best_match.get("authorships", [])
    }


def decode_abstract(inverted_index):
    if not inverted_index:
        return None
    index_to_word = []
    for word, positions in inverted_index.items():
        for pos in positions:
            if len(index_to_word) <= pos:
                index_to_word.extend([None] * (pos - len(index_to_word) + 1))
            index_to_word[pos] = word
    return " ".join(word if word else "[?]" for word in index_to_word)


def fetch_abstracts_for_titles(titles: List[str]) -> List[Dict[str, Any]]:
    """
    Given a list of paper titles, fetch metadata and abstracts from OpenAlex.

    Returns a list of dicts with 'title', 'abstract', and optional metadata.
    Logs how many papers could not be retrieved.
    """
    found = []
    not_found = []

    for title in titles:
        metadata = fetch_openalex_abstract(title)
        if metadata:
            found.append({
                "input_title": title,
                "matched_title": metadata.get("title", ""),
                "abstract": decode_abstract(metadata.get("abstract", {})),
                "id": metadata.get("id", ""),
                "doi": metadata.get("doi", ""),
                "publication_year": metadata.get("publication_year", ""),
                "authors": [auth["author"]["display_name"] for auth in metadata.get("authorships", [])]
            })
        else:
            not_found.append(title)

    logging.info(f"🔍 {len(found)} / {len(titles)} papers found on OpenAlex.")
    if not_found:
        logging.warning(f"⚠️ Could not find {len(not_found)} paper(s):")
        for nf in not_found:
            logging.warning(f" - {nf}")

    return found


def store_openalex_papers_in_vector_db(papers: list, collection = "papers"):
    """
    Embeds and stores OpenAlex papers in the Chroma vector DB.

    Args:
        papers (list): List of OpenAlex paper dicts with abstract and metadata.
        collection: ChromaDB collection handle.
    """
    embedder = OpenAIEmbeddingModel()

    # Filter papers with valid abstracts
    papers_with_abstracts = [p for p in papers if p.get("abstract")]
    if not papers_with_abstracts:
        logger.warning("No valid abstracts found in provided paper list. Skipping embedding.")
        return

    logger.info(f"Preparing to embed {len(papers_with_abstracts)} papers.")

    # Embed abstracts
    abstracts = [p["abstract"] for p in papers_with_abstracts]
    embeddings = embedder.embed_texts(abstracts)

    # Prepare items for ChromaDB
    items_to_store = [
    {
        "id": paper["id"],
        "document": paper["abstract"],
        "embedding": embedding,
        "metadata": {
            "input_title": paper["input_title"],
            "matched_title": paper["matched_title"],
            "doi": paper["doi"],
            "publication_year": paper["publication_year"],
            "authors": ", ".join(paper["authors"]),  # ✅ Fix: flatten list to string
        },
    }
    for paper, embedding in zip(papers, embeddings)
]

    logger.info(f"Storing {len(items_to_store)} embedded papers into ChromaDB.")
    store_multiple_items(collection, items_to_store)
    logger.info("Successfully stored papers in ChromaDB.")


if __name__ == "__main__":
    # Execute in terminal with: "python -m data_loader.openalex_loader"
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 2️⃣ Initiate Chroma DB and collection
    db_dir = "./chroma_db"
    client = initiate_chroma_db(db_dir)
    collection = client.get_or_create_collection("papers")

    # 1. Define a list of paper titles to test
    paper_titles = [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "A Survey on Graph Neural Networks"
    ]

    logger.info("Fetching OpenAlex metadata and abstracts...")
    papers = fetch_abstracts_for_titles(paper_titles)

    if not papers:
        logger.warning("No papers retrieved from OpenAlex. Exiting.")

    # 3. Embed and store in vector DB
    logger.info("Storing papers into ChromaDB...")
    store_openalex_papers_in_vector_db(papers, collection)

    logger.info("Done.")
