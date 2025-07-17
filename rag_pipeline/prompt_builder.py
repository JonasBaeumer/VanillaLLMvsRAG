"""
Prompt builder utilities for the RAG pipeline.

This module provides functions to construct prompts for LLM-based review generation,
including context injection and review guideline formatting.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def build_review_prompt(paper: dict, guidelines: str, sample_reviews: str, context_chunks: List[str]) -> str:
    """
    Build a peer review prompt for an academic paper, including guidelines, paper content,
    retrieved context, and an example review.

    Parameters
    ----------
    paper : dict
        Dictionary with keys 'title', 'abstract', and 'full_text'.
    guidelines : str
        Review guidelines to be followed.
    sample_reviews : str
        Example review for format and tone.
    context_chunks : List[str]
        List of additional context strings retrieved from similar papers.

    Returns
    -------
    str
        The formatted prompt string for the LLM.
    """
    if any(chunk is None for chunk in context_chunks):
        print("⚠️ Warning: One or more context chunks were None and replaced with an empty string.")

    context_text = "\n\n".join(chunk or "" for chunk in context_chunks)

    return f"""
    You are a peer reviewer for an academic conference. Your task is to write a constructive, well-structured peer review based on the provided paper, using the journal’s review guidelines and the example review for inspiration.

    ---

    🧾 Below are the review guidelines (follow these when writing your review):
    {guidelines}

    ---

    📄 Here is the paper:
    Title: {paper['title']}
    Abstract: {paper['abstract']}
    Full Text: {paper['full_text']}

    ---

    📚 Additional Related Context Chunks (retrieved from similar papers):
    {context_text}
    
    ---
    
    📝 Example Review (for format and tone only — unrelated to this paper):
    {sample_reviews}

    Please return ONLY a valid JSON object with the following structure and field names exactly:

    "paper_summary": "...",
    "summary_of_strengths": "...",
    "summary_of_weaknesses": "...",
    "comments_suggestions_and_typos": "...",
    "scores": {{
        "soundness": 0-5,
        "overall_assessment": 0-5
    }}
    

    Do not include any extra commentary, markdown formatting, or explanation. Only return the raw JSON block.
    """


def build_prompt(user_query: str, context_chunks: List[str], template: str) -> List[Dict[str, str]]:
    """
    Builds a prompt using the given template.

    Parameters:
    - user_query: The user question.
    - context_chunks: List of retrieved context strings (can be empty for LLM-only).
    - template: Prompt template string with placeholders {context} and {question}.

    Returns:
    - List of chat messages [{role, content}, ...]
    """
    # Join context chunks into 1 string
    context_text = "\n\n".join(context_chunks) if context_chunks else ""

    formatted_prompt = template.format(context=context_text, question=user_query)

    messages = [
        {"role": "system", "content": formatted_prompt}
    ]

    logger.info("Built prompt with %d context chunks.", len(context_chunks))
    return messages