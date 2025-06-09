# rag_pipeline/generator.py

import logging
from typing import List, Dict
from models.base import LLM

logger = logging.getLogger(__name__)

def generate_answer(llm: LLM, messages: List[Dict[str, str]]) -> str:
    """
    Calls the LLM to generate an answer from the given chat messages.

    Parameters
    ----------
    llm : LLM
        The language model instance.
    messages : List[Dict[str, str]]
        Chat messages [{role: ..., content: ...}].

    Returns
    -------
    str
        The generated answer text.
    """
    try:
        response = llm.generate_text(messages)
        logger.info("Generated answer.")
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."