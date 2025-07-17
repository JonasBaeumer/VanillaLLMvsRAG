"""
Loader and merger utilities for evaluation datasets.

This module provides functions to load JSON datasets, merge LLM-only and RAG review outputs,
and save merged results for evaluation workflows.
"""
import json
from pathlib import Path
from typing import List, Dict

def load_json(path: str) -> List[Dict]:
    """
    Loads a JSON file from the given path.

    Args:
        path: The path to the JSON file.

    Returns:
        A list of dictionaries loaded from the JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_llm_reviews(rag_data: list, llm_data: list) -> list:
    """
    Adds the `llm_generated_review` from LLM-only data to the corresponding paper in RAG data.

    Matching is done via the `paper_id` field.

    Args:
        rag_data: The list of RAG data entries.
        llm_data: The list of LLM-only data entries.

    Returns:
        The updated list of RAG data entries with LLM reviews merged.
    """
    llm_map = {entry["paper_id"]: entry.get("llm_generated_review") for entry in llm_data}

    for paper in rag_data:
        paper_id = paper["paper_id"]
        if paper_id in llm_map:
            paper["llm_generated_review"] = llm_map[paper_id]
        else:
            paper["llm_generated_review"] = {}  # fallback if no match

    return rag_data


def save_json(path: str, data: List[Dict]):
    """
    Saves a list of dictionaries to a JSON file.

    Args:
        path: The path where to save the JSON file.
        data: The list of dictionaries to save.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Paths to your input and output files
    rag_path = "rag_pipeline/output.json"
    llm_only_path = "llm_only_pipeline/output.json"
    # merged_output_path = "evaluation/merged_reviews.json"

    # Step 1: Load both datasets
    rag_data = load_json(rag_path)
    print(f"✅ Loaded {len(rag_data)} entries from RAG output.")

    llm_data = load_json(llm_only_path)
    print(f"✅ Loaded {len(llm_data)} entries from LLM-only output.")

    # Step 2: Merge
    merged_data = merge_llm_reviews(rag_data, llm_data)
    print("🔀 Merged LLM-only reviews into RAG data.")

    # Step 3: Save
    save_json(merged_output_path, merged_data)
    print(f"📁 Saved merged dataset to: {merged_output_path}")

    # Step 4: Print sample comparison
    if merged_data:
        sample = merged_data[0]
        print("\n" + "=" * 100)
        print(f"📄 Title: {sample['metadata'].get('title', '[no title]')}")
        print("=" * 100)

        print("\n🧠 LLM-Only Review:")
        print(json.dumps(sample.get("llm_generated_review", {}), indent=2, ensure_ascii=False))

        print("\n📚 LLM + RAG Review:")
        print(json.dumps(sample.get("llm_plus_rag_generated_review", {}), indent=2, ensure_ascii=False))