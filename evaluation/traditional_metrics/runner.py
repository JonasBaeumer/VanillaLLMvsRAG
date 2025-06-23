import random
import logging
from evaluation.traditional_metrics.bleurt_scorer import BleurtScorer
from evaluation.traditional_metrics.blanc_scorer import BlancScorer

logger = logging.getLogger(__name__)

def extract_review_text(review_obj: dict) -> str:
    """Concatenates all important fields into a single reference string."""
    parts = [
        review_obj.get("topic_and_contributions", ""),
        review_obj.get("reasons_to_accept", ""),
        review_obj.get("reasons_to_reject", ""),
        review_obj.get("questions_for_authors", ""),
        review_obj.get("missing_references", ""),
        review_obj.get("typos_and_style", ""),
        review_obj.get("ethical_concerns", "")
    ]
    return " ".join(part.strip() for part in parts if part.strip())


def extract_generated_review_text(review_obj: dict) -> str:
    """Flattens structured generated review dict into a string for scoring."""
    return " ".join(
        str(value).strip() for key, value in review_obj.items()
        if isinstance(value, str) and value.strip()
    )

def run_traditional_metrics(dataset):
    logger.info("Starting traditional metrics evaluation (BLANC, BLEURT)...")

    bleurt = BleurtScorer()
    blanc = BlancScorer()

    for entry in dataset:
        all_reviews = entry.get("reviews", [])
        if not all_reviews:
            logger.warning("No human reviews found for entry. Skipping.")
            continue

        selected_review = random.choice(all_reviews)
        reference = extract_review_text(selected_review)

        llm_review = extract_generated_review_text(entry["llm_generated_review"])
        rag_review = extract_generated_review_text(entry["llm_plus_rag_generated_review"])

        entry["scores"] = {
            "llm_review": {
                "bleurt": bleurt.score([reference], [llm_review])[0],
                "blanc": blanc.score([reference], [llm_review])[0]
            },
            "rag_review": {
                "bleurt": bleurt.score([reference], [rag_review])[0],
                "blanc": blanc.score([reference], [rag_review])[0]
            }
        }

    # Optional save
    # save_json(dataset, "traditional_metrics_results.json")
    logger.info("Scoring completed.")
    return dataset
    