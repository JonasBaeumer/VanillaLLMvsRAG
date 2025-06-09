import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

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