import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def build_review_prompt(paper: dict, guidelines: str, sample_reviews: str) -> str:
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

def build_prompt(user_query: str, template: str) -> List[Dict[str, str]]:
    """
    Builds a prompt using the given template for LLM-only baseline.

    Parameters
    ----------
    user_query: The user question.
    template: Prompt template string with placeholder {question}.

    Returns
    -------
    List of chat messages [{role, content}, ...]
    """
    formatted_prompt = template.format(question=user_query)

    messages = [
        {"role": "system", "content": formatted_prompt}
    ]

    logger.info("Built LLM-only prompt.")
    return messages