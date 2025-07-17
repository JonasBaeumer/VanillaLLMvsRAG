"""
BLEURT scorer wrapper for traditional metrics evaluation.

This module provides a class to compute BLEURT scores for review evaluation.
"""
from bleurt import score as bleurt_score
import os
import logging

logger = logging.getLogger(__name__)
BLEURT_CHECKPOINT = os.getenv(
    "BLEURT_CHECKPOINT",
    os.path.abspath("bleurt_checkpoints/BLEURT-20")
)

class BleurtScorer:
    """
    BLEURT scorer for evaluating the similarity between reference and generated reviews.
    """
    def __init__(self):
        """
        Initialize the BLEURT scorer with the specified checkpoint.
        """
        logger.info(f"Initializing BLEURT scorer with checkpoint: {BLEURT_CHECKPOINT}")
        self.scorer = bleurt_score.BleurtScorer(checkpoint=BLEURT_CHECKPOINT)

    def score(self, references: list[str], candidates: list[str]) -> list[float]:
        """
        Compute BLEURT scores for each (reference, candidate) pair.

        Args:
            references (list of str): Reference texts.
            candidates (list of str): Generated texts to score.

        Returns:
            list of float: BLEURT scores for each pair.
        """
        logger.info(f"Scoring {len(references)} (reference, candidate) pairs with BLEURT")
        scores = self.scorer.score(references=references, candidates=candidates)
        logger.info(f"Scoring complete. Returning BLEURT scores.")
        return scores

# if __name__ == "__main__":
#     # LOCAL TESTING ONLY: The following block is for manual/local testing and should not be run in production or on import.
#     logging.basicConfig(level=logging.INFO)
#     scorer = BleurtScorer()
#     refs = ["This paper explores the impact of co-occurrence bias in LLMs."]
#     gens = ["The study investigates how co-occurrence influences factual recall in language models."]
#     scores = scorer.score(refs, gens)
#     print(f"BLEURT Score: {scores[0]}")