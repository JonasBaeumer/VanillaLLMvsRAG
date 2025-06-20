import logging
from models.openai_models import OpenAILLM
import json
import re
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

    # Step 2: Generate LLM reviews for each paper
    for paper in dataset:
        doc = paper["docling_paper"]

        if not paper["metadata"].get("abstract"):
            logger.warning(f"Skipping paper {paper['paper_id']} without abstract.")
            continue

        prompt = build_review_prompt(
            paper={
                "title": paper["metadata"].get("title", "[no title]"),
                "abstract": paper["metadata"].get("abstract", ""),
                "full_text": doc.get("full_text", "")
            },
            guidelines=review_guidelines,
            sample_reviews=sample_paper_reviews
        )

        # Generate and store review
        try:
            raw_review = llm.generate_text([
                {"role": "system", "content": "You are an academic peer reviewer."},
                {"role": "user", "content": prompt}
            ])

            # Remove Markdown code block if present
            if raw_review.strip().startswith("```json"):
                raw_review = re.sub(r"^```json\s*", "", raw_review.strip())  # remove leading ```json
                raw_review = re.sub(r"\s*```$", "", raw_review.strip()) # remove trailing ```
            
            # UNCOMMENT FOR DEBUGGING
            # print("\n📦 Raw LLM Output:")
            # print(raw_review)

            parsed_review = json.loads(raw_review)
            paper["llm_generated_review"] = parsed_review

        except Exception as e:
            logger.error(f"Failed to generate review for {paper['paper_id']}: {e}")
            paper["llm_generated_review"] = None

    # Step 3: Pretty print human vs LLM reviews
    for paper in dataset:
        doc = paper["docling_paper"]
        title = paper["metadata"].get("title", "[No Title]")
        human_reviews = paper["reviews"]
        llm_review = paper["llm_generated_review"]

        print("\n" + "=" * 80)
        print(f"📄 Title: {title}")
        print("-" * 80)

        if human_reviews:
            print("🧑 Human Review (first):")
            print(json.dumps(human_reviews[0], indent=2))
        else:
            logger.error("Human Review: Not available")

        print("\n🤖 LLM-Generated Review:")
        if llm_review:
            print(json.dumps(llm_review, indent=2))
        else:
            logger.error("Failed to generate review.")
        print("=" * 80)


if __name__ == "__main__":
    main()