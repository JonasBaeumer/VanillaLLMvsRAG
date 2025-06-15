# llm_only_pipeline/run_llm_only.py

import logging
from models.openai_models import OpenAILLM
from llm_only_pipeline.prompt_templates import LLM_ONLY_TEMPLATE_V1
from llm_only_pipeline.prompt_builder import build_prompt
from models.generator import generate_answer 
from sample_papers import sample_paper_one_abstract, sample_paper_one_main, sample_paper_one_references, sample_paper_one_reviews, one_shot_review_example
from acl_review_guidelines import review_guidelines

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # Step 0: Init LLM model
    llm = OpenAILLM()

    # Step 1: Define user query
    user_query = f""" 
    You are a peer reviewer for an academic journal. Your task is to write a constructive, well-structured peer review based on the provided paper, using the journal’s review guidelines and the example review for inspiration.

    ---

    📄 PAPER TO REVIEW:
    {sample_paper_one_main}

    ---
    
    📄 PAPER ABSTRACT:
    {sample_paper_one_abstract}

    ---

    📝 EXAMPLE REVIEW (for format and tone only — unrelated to this paper):
    {one_shot_review_example}

    ---

    📋 REVIEW GUIDELINES (follow these when writing your review):
    {review_guidelines}

    ---

    ✍️ Now write your review of the paper. Be concise but thorough. Cover key areas such as relevance, novelty, methodology, clarity, and impact. Structure your feedback into clear sections or bullet points if appropriate. Use a neutral, professional tone. Avoid repeating the example content — it’s for format only.
    """

    # Step 2: Build prompt (LLM-only)
    messages = build_prompt(user_query, template=LLM_ONLY_TEMPLATE_V1)

    # Step 3: Generate answer from LLM
    answer = generate_answer(llm, messages)

    # Step 4: Output results
    print("\n🔍 User Query:", user_query)
    print("\n🧠 LLM-Only Answer:\n", answer)

if __name__ == "__main__":
    main()