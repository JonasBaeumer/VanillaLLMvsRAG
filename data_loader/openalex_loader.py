import requests
import time
from urllib.parse import quote
import logging
from typing import List, Dict, Any
from data_loader.constants import OPENALEX_WORKS_ENDPOINT
from models.openai_models import OpenAIEmbeddingModel
from chroma_db.chroma import store_multiple_items, initiate_chroma_db
from data_loader.arxiv_loader import fetch_and_chunk_arxiv_full_text
import logging
import os

logger = logging.getLogger("openalex-loader")
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OPENALEX_API_KEY = os.getenv("OPEN_ALEX_PREMIUM_API_KEY")

def fetch_openalex_abstract(title, delay=1.0):
    """
    Queries OpenAlex with a paper title and returns metadata including the abstract.
    """
    query_url = f"{OPENALEX_WORKS_ENDPOINT}?search={quote(title)}&api_key={OPENALEX_API_KEY}"

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
        "primary_location": best_match.get("primary_location", {}),
        "open_access": best_match.get("open_access", {}),
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

    If possible, retrieve the full text from arxiv based on the rectrieved OpenAlex metadata.

    Returns a list of dicts with 'title', 'abstract', and optional metadata.
    Logs how many papers could not be retrieved.
    """
    found = []
    not_found = []
    found_full_text = []

    for title in titles:
        metadata = fetch_openalex_abstract(title)
        if metadata:

            full_text_chunked = []

            # Check if the paper is available on arXiv
            if metadata['primary_location']['source'].get("display_name") == "arXiv (Cornell University)":
                oa_url = metadata['open_access'].get('oa_url', {})
                # Check if the OA URL is available
                if oa_url:
                    full_text_chunked = fetch_and_chunk_arxiv_full_text(oa_url)
                    found_full_text.append(title)

            found.append({
                "input_title": title,
                "matched_title": metadata.get("title", ""),
                "abstract": decode_abstract(metadata.get("abstract", {})),
                "full_text_chunked": full_text_chunked,
                "id": metadata.get("id", ""),
                "doi": metadata.get("doi", ""),
                "publication_year": metadata.get("publication_year", ""),
                "authors": [auth["author"]["display_name"] for auth in metadata.get("authorships", [])]
            })
        else:
            not_found.append(title)

    logging.info(f"🔍 {len(found)} / {len(titles)} papers found on OpenAlex.")
    logging.info(f"📄 {len(found_full_text)} / {len(found)} found papers papers have full text available on arXiv.")
    if not_found:
        logging.warning(f"⚠️ Could not find {len(not_found)} paper(s):")
        for nf in not_found:
            logging.warning(f" - {nf}")

    return found


def store_openalex_papers_in_vector_db(papers: list, collection_name = "papers"):
    """
    Embeds and stores OpenAlex papers in the Chroma vector DB.

    Args:
        papers (list): List of OpenAlex paper dicts with abstract and metadata.
        collection_name (str): Name of the ChromaDB collection to store papers in.
    """
    embedder = OpenAIEmbeddingModel()

    chroma = initiate_chroma_db("./chroma_db")
    collection = chroma.get_or_create_collection(collection_name)

    items_to_store = []

    for paper in papers:
        paper_id = paper["id"]
        base_metadata = {
            "input_title": paper.get("input_title") or "",
            "matched_title": paper.get("matched_title") or "",
            "doi": paper.get("doi") or "",
            "publication_year": paper.get("publication_year") or -1,
            "authors": ", ".join(paper.get("authors") or []),
        }
        # If full text chunks exist, embed and store each chunk
        if paper.get("full_text_chunked"):
            chunks = paper["full_text_chunked"]
            embeddings = embedder.embed_texts(chunks)
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                items_to_store.append({
                    "id": f"{paper_id}_chunk_{idx}",
                    "document": chunk,
                    "embedding": embedding,
                    "metadata": {**base_metadata, "chunk_index": idx},
                })
        # Otherwise, embed and store the abstract
        elif paper.get("abstract"):
            embedding = embedder.embed_texts([paper["abstract"]])[0]
            items_to_store.append({
                "id": f"{paper_id}_abstract",
                "document": paper["abstract"],
                "embedding": embedding,
                "metadata": {**base_metadata, "chunk_index": -1},
            })

    if not items_to_store:
        logger.warning("No valid documents found in provided paper list. Skipping embedding.")
        return

    logger.info(f"Storing {len(items_to_store)} embedded documents into ChromaDB.")
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
    store_openalex_papers_in_vector_db(papers)

    logger.info("Done.")
