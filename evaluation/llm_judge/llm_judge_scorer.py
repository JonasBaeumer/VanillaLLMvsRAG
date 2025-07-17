"""
LLM judge scoring logic for ELO-based evaluation.

This module provides the function to run the LLM-based ELO judging on a dataset using the EloEngine.
"""
import logging
from .elo_engine import EloEngine

logger = logging.getLogger(__name__)

def run_llm_judge_scoring(dataset, num_rounds=10):
    """
    Runs the LLM-based ELO judging on a dataset.

    Args:
        dataset (list): The list of entries to judge.
        num_rounds (int): The number of rounds to run for the tournament.

    Returns:
        dict: A dictionary containing the ELO scores and rounds.
    """
    logger.info(f"Starting LLM-based ELO judging for {len(dataset)} entries with {num_rounds} rounds...")

    engine = EloEngine()
    elo_result = engine.run_tournament(dataset, num_rounds=num_rounds)

    logger.info("ELO-based judging complete.")
    return {
        "elo_scores": elo_result["elo_scores"],
        "elo_rounds": elo_result["elo_rounds"]
    }