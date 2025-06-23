from bleurt import score as bleurt_score
import os
import logging

logger = logging.getLogger(__name__)
BLEURT_CHECKPOINT = os.getenv(
    "BLEURT_CHECKPOINT",
    os.path.abspath("bleurt_checkpoints/BLEURT-20")
)

class BleurtScorer:
    def __init__(self):
        logger.info(f"Initializing BLEURT scorer with checkpoint: {BLEURT_CHECKPOINT}")
        self.scorer = bleurt_score.BleurtScorer(checkpoint=BLEURT_CHECKPOINT)

    def score(self, references: list[str], candidates: list[str]) -> list[float]:
        logger.info(f"Scoring {len(references)} (reference, candidate) pairs with BLEURT")
        scores = self.scorer.score(references=references, candidates=candidates)
        logger.info(f"Scoring complete. Returning BLEURT scores.")
        return scores

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scorer = BleurtScorer()
    refs = ["This paper explores the impact of co-occurrence bias in LLMs."]
    gens = ["The study investigates how co-occurrence influences factual recall in language models."]
    scores = scorer.score(refs, gens)
    print(f"BLEURT Score: {scores[0]}")