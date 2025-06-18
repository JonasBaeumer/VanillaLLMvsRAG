import json
from pathlib import Path


def extract_docling_sections(docling_data):
    furniture = docling_data.get("furniture", {})
    sections = []
    full_text_parts = []

    for sec in furniture.get("sections", []):
        heading = sec.get("heading", "")
        text = sec.get("text", "")
        sections.append({"heading": heading, "text": text})
        if heading:
            full_text_parts.append(f"{heading}\n{text}")
        else:
            full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)
    return {
        "title": furniture.get("title", "[no title]"),
        "authors": furniture.get("authors", []),
        "sections": sections,
        "full_text": full_text,
    }


def format_reviews(reviews_data):
    formatted_reviews = []
    for review in reviews_data:
        r = review.get("report", {})
        scores = review.get("scores", {})
        meta = review.get("meta", {})

        formatted_reviews.append({
            "reviewer_id": review.get("rid", "[unknown]"),
            "topic_and_contributions": r.get("paper_topic_and_main_contributions", ""),
            "reasons_to_accept": r.get("reasons_to_accept", ""),
            "reasons_to_reject": r.get("reasons_to_reject", ""),
            "questions_for_authors": r.get("questions_for_the_authors", ""),
            "missing_references": r.get("missing_references", ""),
            "typos_and_style": r.get("typos_grammar_style_and_presentation_improvements", ""),
            "ethical_concerns": r.get("ethical_concerns", ""),
            "scores": {
                "soundness": scores.get("soundness", ""),
                "excitement": scores.get("excitement", ""),
                "reproducibility": scores.get("reproducibility", "")
            },
            "reviewer_confidence": meta.get("reviewer_confidence", "")
        })
    return formatted_reviews


def load_arr_emnlp_dataset(base_path):
    root = Path(base_path)
    dataset = []

    for paper_dir in (root / "data").iterdir():
        if not paper_dir.is_dir():
            continue

        paper_id = paper_dir.name
        paper_data = {"paper_id": paper_id, "metadata": {}, "docling_paper": {}, "reviews": []}

        # Load paper-level metadata
        paper_meta = paper_dir / "meta.json"
        if paper_meta.exists():
            with open(paper_meta) as f:
                paper_data["metadata"] = json.load(f)

        # Always use v1 folder only for now
        v1_dir = paper_dir / "v1"
        if not v1_dir.exists():
            continue

        # Load docling version of the paper
        docling_path = v1_dir / "paper.docling.json"
        if docling_path.exists():
            with open(docling_path) as f:
                docling_data = json.load(f)
                paper_data["docling_paper"] = extract_docling_sections(docling_data)

        # Load reviews
        reviews_path = v1_dir / "reviews.json"
        if reviews_path.exists():
            with open(reviews_path) as f:
                reviews_data = json.load(f)
                paper_data["reviews"] = format_reviews(reviews_data)

        dataset.append(paper_data)

    return dataset