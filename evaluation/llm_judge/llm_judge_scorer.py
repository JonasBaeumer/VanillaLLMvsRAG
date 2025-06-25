import logging
from .elo_engine import EloEngine

logger = logging.getLogger(__name__)

def run_llm_judge_scoring(dataset, num_rounds=10):
    logger.info(f"Starting LLM-based ELO judging for {len(dataset)} entries with {num_rounds} rounds...")

    engine = EloEngine()
    elo_result = engine.run_tournament(dataset, num_rounds=num_rounds)

    logger.info("ELO-based judging complete.")
    return {
        "elo_scores": elo_result["elo_scores"],
        "elo_rounds": elo_result["elo_rounds"]
    }