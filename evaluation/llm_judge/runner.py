import logging
from .llm_judge_scorer import run_llm_judge_scoring

logger = logging.getLogger(__name__)

def run_llm_judge_pipeline(dataset, num_rounds=10):
    """
    Runs the LLM-based ELO judging on the provided dataset.

    Parameters:
    - dataset: list of merged paper entries (with human, LLM, and RAG reviews)
    - num_rounds: number of random comparison rounds to perform in the tournament

    Returns:
    - dict with:
        - "data": original dataset (unchanged)
        - "elo_evaluation": {
            "elo_scores": final ELO scores per system,
            "elo_rounds": round-wise logs
        }
    """
    logger.info("🔍 Running LLM Judge Pipeline...")
    elo_result = run_llm_judge_scoring(dataset, num_rounds=num_rounds)

    return {
        "data": dataset,  # Dataset remains unchanged
        "elo_evaluation": elo_result
    }