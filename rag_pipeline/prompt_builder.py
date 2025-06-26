# rag_pipeline/prompt_builder.py

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def build_review_prompt(paper: dict, guidelines: str, sample_reviews: str, context_chunks: List[str]) -> str:
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

    Now, please write a review following the guidelines. Return the review in JSON format with fields:
    - paper_summary
    - summary_of_strenghts
    - summary_of_weaknesses
    - comments_suggestions_and_typos
    - scores : {{
        "soundess": 0-5,
        "overall_assessment": 0-5,
    }}
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