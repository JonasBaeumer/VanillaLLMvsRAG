# llm_only_pipeline/run_llm_only.py

import logging
from models.openai_models import OpenAILLM
from llm_only_pipeline.prompt_templates import LLM_ONLY_TEMPLATE_V1
from llm_only_pipeline.prompt_builder import build_prompt
from models.generator import generate_answer 

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # Step 0: Init LLM model
    llm = OpenAILLM()

    # Step 1: Define user query
    user_query = "What are the advantages of using retrieval augmented generation?"

    # Step 2: Build prompt (LLM-only)
    messages = build_prompt(user_query, template=LLM_ONLY_TEMPLATE_V1)

    # Step 3: Generate answer from LLM
    answer = generate_answer(llm, messages)

    # Step 4: Output results
    print("\n🔍 User Query:", user_query)
    print("\n🧠 LLM-Only Answer:\n", answer)

if __name__ == "__main__":
    main()