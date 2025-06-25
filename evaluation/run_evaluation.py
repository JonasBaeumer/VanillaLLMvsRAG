import json
import logging
import os
from evaluation.loader import load_json, merge_llm_reviews
from evaluation.traditional_metrics.runner import run_traditional_metrics
from evaluation.llm_judge.runner import run_llm_judge_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Paths to input files
    rag_path = "rag_pipeline/output.json"
    llm_only_path = "llm_only_pipeline/output.json"

    # Step 1: Load
    rag_data = load_json(rag_path)
    print(f"✅ Loaded {len(rag_data)} entries from RAG output.")

    llm_data = load_json(llm_only_path)
    print(f"✅ Loaded {len(llm_data)} entries from LLM-only output.")

    # Step 2: Merge
    merged_data = merge_llm_reviews(rag_data, llm_data)
    print("✅ Merged LLM-only reviews into RAG data.")

    # Step 3: Traditional Metrics Scoring
    scored_data = run_traditional_metrics(merged_data)
    print("✅ Traditional metrics scoring completed.")

    results = run_llm_judge_pipeline(merged_data, num_rounds=1)

    os.makedirs("evaluation", exist_ok=True)

    with open("evaluation/dataset_with_traditional_scores.json", "w", encoding="utf-8") as f:
        json.dump(results["data"], f, indent=2, ensure_ascii=False)

    logger.info("✅ Dataset saved to evaluation/dataset_with_traditional_scores.json")

    with open("evaluation/elo.json", "w", encoding="utf-8") as f:
        json.dump(results["elo_evaluation"], f, indent=2, ensure_ascii=False)

    logger.info("✅ Dataset saved to evaluation/elo.json")

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