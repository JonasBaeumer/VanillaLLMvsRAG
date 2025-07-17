import json
from pathlib import Path
from typing import List, Dict

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_llm_reviews(rag_data: list, llm_data: list) -> list:
    """
    Adds the `llm_generated_review` from LLM-only data to the corresponding paper in RAG data.

    Matching is done via the `paper_id` field.
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# if __name__ == "__main__":
#     LOCAL TESTING ONLY: The following block is for manual/local testing and should not be run in production or on import.
#     rag_path = "rag_pipeline/output.json"
#     llm_only_path = "llm_only_pipeline/output.json"
#     rag_data = load_json(rag_path)
#     print(f"✅ Loaded {len(rag_data)} entries from RAG output.")
#     llm_data = load_json(llm_only_path)
#     print(f"✅ Loaded {len(llm_data)} entries from LLM-only output.")
#     merged_data = merge_llm_reviews(rag_data, llm_data)
#     print("🔀 Merged LLM-only reviews into RAG data.")
#     if merged_data:
#         sample = merged_data[0]
#         print("\n" + "=" * 100)
#         print(f"📄 Title: {sample['metadata'].get('title', '[no title]')}")
#         print("=" * 100)
#         print("\n🧠 LLM-Only Review:")
#         print(json.dumps(sample.get("llm_generated_review", {}), indent=2, ensure_ascii=False))
#         print("\n📚 LLM + RAG Review:")
#         print(json.dumps(sample.get("llm_plus_rag_generated_review", {}), indent=2, ensure_ascii=False))