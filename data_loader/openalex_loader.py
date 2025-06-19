import requests
import time
from urllib.parse import quote
import logging
from typing import List, Dict, Any
from data_loader.constants import OPENALEX_WORKS_ENDPOINT


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

if __name__ == "__main__":
    # Execute in terminal with: "python -m data_loader.openalex_loader"
    paper_titles = [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "Some Random Nonexistent Title 123456"
    ]

    results = fetch_abstracts_for_titles(paper_titles)

    for paper in results:
        print("\n🔍", paper["input_title"])
        print("📄", paper["matched_title"])
        print("📝 Abstract:", paper["abstract"])