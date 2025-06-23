import logging
import torch
from blanc import BlancHelp
import nltk
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
nltk.download("punkt", quiet=True)

# Monkey patch BLANC if it's trying to load non-existent 'punkt_tab'
from nltk import tokenize
tokenize.PunktTokenizer = lambda *args, **kwargs: PunktSentenceTokenizer()

logger = logging.getLogger(__name__)


class BlancScorer:
    def __init__(self, model_name="bert-base-uncased", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing BLANC scorer with model: {model_name} on device: {device}")
        self.model = BlancHelp(model_name=model_name, device=device)

    def score(self, references: list[str], candidates: list[str]) -> list[float]:
        """
        Calculates BLANC-help scores for each (reference, candidate) pair.
        Args:
            references (list[str]): The document (paper abstract or full text).
            candidates (list[str]): The generated reviews.
        Returns:
            list[float]: BLANC-help scores.
        """
        logger.info(f"Scoring {len(candidates)} reviews with BLANC-help.")
        scores = self.model.eval_pairs(docs=references, summaries=candidates)
        logger.info(f"Scoring complete.")
        return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scorer = BlancScorer()

    doc = (
        "This paper introduces a new method for improving factual consistency in large language models. "
        "The authors propose a co-occurrence-based strategy to select fine-tuning data. "
        "Experimental results on the LAMA benchmark demonstrate improved performance."
    )

    review = (
        "The paper proposes a novel method using co-occurrence statistics to improve factual consistency. "
        "The results on the LAMA dataset show the approach is effective."
    )

    score = scorer.score([doc], [review])
    print(f"BLANC-help score: {score[0]:.4f}")