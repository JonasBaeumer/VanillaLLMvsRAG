import json
from pathlib import Path
from models.openai_models import OpenAILLM
from .utils import (
    convert_docling_json_to_markdown,
    split_markdown_sections,
    get_title_and_authors_from_furniture,
    extract_titles_with_llm)

llm = OpenAILLM(model_name="gpt-4o")

def extract_docling_paper(docling_data: dict) -> dict:
    """Convert a Docling JSON paper to a simple dict.

    The dict now contains **only**:
        - title              (str)
        - authors            (list[str])
        - full_text          (str, markdown for the *whole* paper)
        - references_markdown(str, markdown text of the References section)
    """

    markdown = convert_docling_json_to_markdown(docling_data).strip()
    sections = split_markdown_sections(markdown)

    # --- Front‑matter -------------------------------------------------------
    title, authors = get_title_and_authors_from_furniture(docling_data)

    # --- Pull out the References section -----------------------------------
    references_parts: list[str] = []
    for sec in sections:
        heading = (sec.get("heading") or "").lower()
        if "reference" in heading:  # catch “Reference”, “References”, etc.
            references_parts.append(sec.get("text", ""))

    references_markdown = "\n".join(references_parts).strip()

    return {
        "title": title,
        "authors": authors,
        "full_text": markdown,
        "references_markdown": references_markdown,
    }


# ---------------------------------------------------------------------------
# Reviews helpers (unchanged except for minor style tweaks)
# ---------------------------------------------------------------------------

def format_reviews(reviews_data):
    formatted_reviews = []
    for review in reviews_data:
        r = review.get("report", {})
        scores = review.get("scores", {})
        meta = review.get("meta", {})

        formatted_reviews.append(
            {
                "reviewer_id": review.get("rid", "[unknown]"),
                "topic_and_contributions": r.get("paper_topic_and_main_contributions", ""),
                "reasons_to_accept": r.get("reasons_to_accept", ""),
                "reasons_to_reject": r.get("reasons_to_reject", ""),
                "questions_for_authors": r.get("questions_for_the_authors", ""),
                "missing_references": r.get("missing_references", ""),
                "typos_and_style": r.get(
                    "typos_grammar_style_and_presentation_improvements", ""
                ),
                "ethical_concerns": r.get("ethical_concerns", ""),
                "scores": {
                    "soundness": scores.get("soundness", ""),
                    "excitement": scores.get("excitement", ""),
                    "reproducibility": scores.get("reproducibility", ""),
                },
                "reviewer_confidence": meta.get("reviewer_confidence", ""),
            }
        )
    return formatted_reviews


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_arr_emnlp_dataset(base_path, llm: OpenAILLM = None, rag_eval=False):
    root = Path(base_path)
    dataset = []

    for paper_dir in root.iterdir():
        if not paper_dir.is_dir():
            continue

        paper_id = paper_dir.name
        paper_data = {
            "paper_id": paper_id,
            "metadata": {},
            "docling_paper": {},
            "reviews": [],
        }

        v1_dir = paper_dir / "v1"  # always use v1 for now
        if not v1_dir.exists():
            continue

        # -------- meta.json --------------------------------------------------
        v1_meta = v1_dir / "meta.json"
        if v1_meta.exists() and v1_meta.stat().st_size:
            try:
                paper_data["metadata"] = json.loads(v1_meta.read_text())
            except json.JSONDecodeError:
                print(f"⚠️  Skipping malformed meta.json in {paper_id}")
        else:
            print(f"⚠️  No usable meta.json in v1 for paper {paper_id}")

        # -------- paper.docling.json ----------------------------------------
        docling_path = v1_dir / "paper.docling.json"
        if docling_path.exists() and docling_path.stat().st_size:
            try:
                paper_data["docling_paper"] = extract_docling_paper(
                    json.loads(docling_path.read_text())
                )
            except json.JSONDecodeError:
                print(f"⚠️  Skipping malformed docling JSON for {paper_id}")

        # -------- reviews.json ----------------------------------------------
        reviews_path = v1_dir / "reviews.json"
        if reviews_path.exists() and reviews_path.stat().st_size:
            try:
                paper_data["reviews"] = format_reviews(
                    json.loads(reviews_path.read_text())
                )
            except json.JSONDecodeError:
                print(f"⚠️  Skipping malformed reviews.json for {paper_id}")

        # -------- Extract reference titles with LLM (only for RAG evaluation) --------

        if rag_eval:
            reference_block = paper_data["docling_paper"].get("references_markdown", "")
            if reference_block:
                titles = extract_titles_with_llm(reference_block, model=llm)
                paper_data["docling_paper"]["reference_titles"] = titles

        dataset.append(paper_data)

    return dataset


# ---------------------------------------------------------------------------
# Quick CLI for sanity‑checking
# ---------------------------------------------------------------------------

def main():

    from textwrap import shorten
    dataset_path = "./data/ARR-EMNLP"  # Adjust as needed
    dataset = load_arr_emnlp_dataset(dataset_path, llm, True)

    print(f"✅ Loaded {len(dataset)} papers.\n")

    for i, paper in enumerate(dataset[:2]):
        doc = paper["docling_paper"]
        title = doc.get("title", "[no title]")
        authors = ", ".join(doc.get("authors", [])) or "[no authors]"
        full_text_preview = shorten(doc.get("full_text", ""), width=120, placeholder=" …")
        refs_preview = shorten(doc.get("references_markdown", ""), width=300, placeholder=" …")
        ref_list = doc.get("reference_titles", "")

        print(f"\n📄 Paper #{i+1} — ID: {paper['paper_id']}")
        print("🔹 Title:", title)
        print("🔹 Authors:", authors)
        print("🔹 Full‑text preview:", full_text_preview)
        print("🔹 References preview:", refs_preview or "[none]")
        print("🔹 References list:", ref_list or "[none]")

        if paper["reviews"]:
            print("\n📝 Reviews:")
            for j, review in enumerate(paper["reviews"]):
                print(f"  🔸 Review #{j+1} — Reviewer ID: {review['reviewer_id']}")
                print(
                    "    📌 Topic and Contributions:",
                    shorten(review["topic_and_contributions"], width=100, placeholder=" …"),
                )
                print("    🧪 Scores:", review["scores"])
        else:
            print("\n📝 Reviews:  None found.")


if __name__ == "__main__":
    main()