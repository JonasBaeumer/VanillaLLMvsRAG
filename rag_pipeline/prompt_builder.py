# rag_pipeline/prompt_builder.py

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

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