import json
import logging
import os
import time
from evaluation.loader import load_json, merge_llm_reviews
from evaluation.traditional_metrics.runner import run_traditional_metrics
from evaluation.llm_judge.runner import run_llm_judge_pipeline
from evaluation.util import filter_complete_entries
from evaluation.experiments.dataset_integrity_analyser import analyse_dataset_integrity

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    start_time = time.time()
    # Paths to input files
    rag_path = "rag_pipeline/output.json"
    llm_only_path = "llm_only_pipeline/output.json"

    # Step 1: Load
    rag_data = load_json(rag_path)
    logger.info(f"✅ Loaded {len(rag_data)} entries from RAG output.")

    llm_data = load_json(llm_only_path)
    logger.info(f"✅ Loaded {len(llm_data)} entries from LLM-only output.")

    # Step 2: Merge
    merged_data = merge_llm_reviews(rag_data, llm_data)
    logger.info("✅ Merged LLM-only reviews into RAG data.")

    # Step 3: Keep only entries where all required fields are present
    validated_data = filter_complete_entries(merged_data)

    # BIS HIER NOCH 47 entries vorhanden!
    logger.info(f"✅ Filtered data to {len(validated_data)} complete entries.")
    
    # Step 3: Traditional Metrics Scoring
    scored_data = run_traditional_metrics(validated_data)
    print("✅ Traditional metrics scoring completed.")

    results = run_llm_judge_pipeline(validated_data, num_rounds=len(validated_data)*3)

    os.makedirs("evaluation", exist_ok=True)

    with open("evaluation/dataset_with_traditional_scores.json", "w", encoding="utf-8") as f:
        json.dump(results["data"], f, indent=2, ensure_ascii=False)

    logger.info("✅ Dataset saved to evaluation/dataset_with_traditional_scores.json")

    with open("evaluation/elo.json", "w", encoding="utf-8") as f:
        json.dump(results["elo_evaluation"], f, indent=2, ensure_ascii=False)

    logger.info("✅ Dataset saved to evaluation/elo.json")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"✅ Execution of evaluation_pipeline completed in {elapsed_time:.2f} seconds.")
    #"""
    """
    # Step 4: Preview
    if scored_data:
        sample = scored_data[0]
        print("\n" + "=" * 100)
        print(f"📄 Title: {sample['metadata'].get('title', '[no title]')}")
        print("=" * 100)

        print("\n🧠 LLM-Only Review:")
        print(sample['llm_generated_review'])

        print("\n📚 LLM + RAG Review:")
        print(sample['llm_plus_rag_generated_review'])

        print("\n📈 Scores:")
        print(json.dumps(sample['scores'], indent=2, ensure_ascii=False))
    """