"""
LLM-only pipeline runner script.

This script runs the LLM-only review generation pipeline for academic peer review generation.
It loads datasets, builds prompts, generates reviews using an LLM (without retrieval),
and saves the results for further evaluation.
"""
import logging
import json
from json import JSONDecodeError
import os
import re
import time
from llm_only_pipeline.prompt_templates import LLM_ONLY_TEMPLATE_V1
from llm_only_pipeline.prompt_builder import build_review_prompt
from models.openai_models import OpenAILLM
from models.generator import generate_answer 
from sample_papers import sample_paper_reviews
from acl_review_guidelines import review_guidelines
from data_loader.dataset_loader import load_arr_emnlp_dataset
from data_loader.utils import load_existing_outputs, parse_review_json

logger = logging.getLogger("LLM-only-generation-Pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    """
    Main entry point for the LLM-only pipeline.
    Loads data, generates reviews using an LLM, and saves outputs.
    """

    start_time = time.time()
    # Step 0: Init LLM model
    llm = OpenAILLM()

    # Step 1: Load dataset
    dataset = load_arr_emnlp_dataset("./data/ARR-EMNLP", llm=llm, rag_eval=False)
    logger.info(f"✅ Loaded dataset with {len(dataset)} entries.")

    existing_outputs = load_existing_outputs("./llm_only_pipeline/output.json")
    logger.info(f"Loaded {len(existing_outputs)} existing outputs from llm_only_pipeline/output.json.")

    # Only pass down papers that we were able to process
    completed_papers = []

    # Step 2: Generate LLM reviews for each paper
    for paper in dataset:
        doc = paper["tei_data"]

        paper_id = paper["paper_id"]

        if paper_id in existing_outputs:
            logger.info(f"Skipping already processed paper: {paper_id}")
            continue

        else: 

            logger.info(f"Processing paper: {paper_id}")

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

                parsed_review = parse_review_json(raw_review, paper_id)
                paper["llm_generated_review"] = parsed_review
                completed_papers.append(paper)

            except JSONDecodeError as e:
                logger.error(f"❌ Failed to decode JSON for paper {paper_id}: {e}")

            except Exception as e:
                logger.error(f"Failed to generate review for {paper['paper_id']}: {e}")
                paper["llm_generated_review"] = None
    
    """
    # Step 3: Pretty print human vs LLM reviews (only when debugging)
    for paper in completed_papers:
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
    """
    # Step 4: Save dataset to JSON file for further processing

    os.makedirs("llm_only_pipeline", exist_ok=True)

    with open("llm_only_pipeline/output.json", "w", encoding="utf-8") as f:
        json.dump(completed_papers, f, indent=2, ensure_ascii=False)

    logger.info("✅ Dataset saved to rag_pipeline/output.json")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"✅ Execution of rag_pipeline completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()