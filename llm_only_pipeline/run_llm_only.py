import logging
from models.openai_models import OpenAILLM
import json
from llm_only_pipeline.prompt_templates import LLM_ONLY_TEMPLATE_V1
from llm_only_pipeline.prompt_builder import build_review_prompt
from models.generator import generate_answer 
from sample_papers import sample_paper_reviews
from acl_review_guidelines import review_guidelines
from data_loader.dataset_loader import load_arr_emnlp_dataset

logger = logging.getLogger("LLM-only-generation-Pipeline")
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # Step 0: Init LLM model
    llm = OpenAILLM()

    # Step 1: Load dataset
    dataset = load_arr_emnlp_dataset("./data/ARR-EMNLP", llm=llm, rag_eval=False)

    print(f"✅ Loaded dataset with {len(dataset)} entries.")

    # Step 2: Pretty print human vs LLM reviews
    for paper in dataset:
        title = paper.get("docling_paper", {}).get("title", "[No Title]")
        human_reviews = paper.get("reviews", [])
        llm_review = paper.get("llm_generated_review")

        print("\n" + "=" * 80)
        print(f"📄 Title: {title}")
        print("-" * 80)

        if human_reviews:
            print("🧑 Human Review (first):")
            print(json.dumps(human_reviews[0], indent=2))
        else:
            print("🧑 Human Review: Not available")

        print("\n🤖 LLM-Generated Review:")
        if llm_review:
            print(json.dumps(llm_review, indent=2))
        else:
            print("Failed to generate review.")
        print("=" * 80)


if __name__ == "__main__":
    main()