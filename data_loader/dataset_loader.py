"""
ARR-EMNLP dataset loader and TEI/Docling conversion utilities.

This module provides functions to load, parse, and convert academic paper datasets from TEI XML and Docling JSON formats from the ARR-EMNLP dataset.
It also extracts references, metadata, and prepares data for downstream review generation and retrieval tasks.
"""
import json
import os
from pathlib import Path
from models.openai_models import OpenAILLM
import xml.etree.ElementTree as ET
from typing import Dict, List
import logging
import xml.etree.ElementTree as ET
from .utils import (
    split_markdown_sections,
    get_title_and_authors_from_furniture,
    extract_titles_with_llm)

llm = OpenAILLM(model_name="gpt-4o")
logger = logging.getLogger(__name__)


def extract_markdown_from_div(div, ns):
    head = div.find("tei:head", ns)
    paragraphs = div.findall(".//tei:p", ns)

    heading = head.text.strip() if head is not None and head.text else None
    para_texts = [p.text.strip() for p in paragraphs if p.text]

    # if heading:
        # print(f"✅ Found section heading: {heading}")
    # else:
        # print("⚠️  Section has no heading")

    # print(f"📄 Found {len(para_texts)} paragraph(s)")

    markdown_parts = []
    if heading:
        markdown_parts.append(f"## {heading}")
    markdown_parts.extend(para_texts)

    return "\n\n".join(markdown_parts)

def extract_references_from_tei(tree):
    namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}
    refs_md = []

    # Broaden the XPath to pick up all biblStruct in the back matter
    bibl_nodes = tree.findall(".//tei:back//tei:biblStruct", namespace)
    logger.info(f"📚 Found {len(bibl_nodes)} bibliography entries")

    for bibl in bibl_nodes:
        # Authors
        author_elems = bibl.findall(".//tei:author/tei:persName", namespace)
        authors = []
        for pers in author_elems:
            forename = pers.find("tei:forename", namespace)
            surname = pers.find("tei:surname", namespace)
            name = ""
            if forename is not None and forename.text:
                name += forename.text
            if surname is not None and surname.text:
                name += " " + surname.text
            if name:
                authors.append(name.strip())
        author_str = " & ".join(authors) if authors else "Unknown Author"

        # Title
        title_elem = bibl.find(".//tei:title[@level='a']", namespace) or bibl.find(".//tei:title", namespace)
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Untitled"

        # Year
        year_elem = bibl.find(".//tei:date", namespace)
        year = year_elem.attrib.get("when", "") if year_elem is not None else ""

        # Journal/Publisher (if any)
        source = ""
        monogr = bibl.find(".//tei:monogr", namespace)
        if monogr is not None:
            journal_elem = monogr.find(".//tei:title[@level='j']", namespace)
            if journal_elem is not None and journal_elem.text:
                source = journal_elem.text.strip()

        # Compose markdown
        ref_md = f"- **{author_str}** ({year}). *{title}*"
        if source:
            ref_md += f". {source}"
        refs_md.append(ref_md)

    if not refs_md:
        logger.warning("⚠️ No valid references were extracted.")
    else:
        logger.info(f"✅ Extracted {len(refs_md)} reference entries.")
        # for ref in refs_md[:2]:  # Show a preview (debugging)
            # print("📝", ref)

    # Optional: debug back divs
    # back_divs = tree.findall(".//tei:back//tei:div", namespace)
    # for div in back_divs:
        # print("📘 back div type:", div.attrib.get("type"))

    return "\n".join(refs_md)

def tei_to_docklink_style_dict(file_path):
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return {}

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"❌ Failed to parse TEI XML: {e}")
        return {}

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    title_el = root.find(".//tei:titleStmt/tei:title", ns)
    author_els = root.findall(".//tei:author", ns)
    divs = root.findall(".//tei:text//tei:div", ns)
    references_markdown = extract_references_from_tei(root)

    logger.info(f"📌 Found title: {title_el.text if title_el is not None else 'None'}")

    title = title_el.text.strip() if title_el is not None else ""
    authors = []

    all_markdown_sections = []
    references_section = ""

    for i, div in enumerate(divs):
        heading = div.find("tei:head", ns)
        heading_text = heading.text.strip().lower() if heading is not None and heading.text else ""

        # print(f"\n🔹 Processing div #{i+1}: {heading_text if heading_text else '[no heading]'}")

        markdown = extract_markdown_from_div(div, ns)

        if "reference" in heading_text or "bibliography" in heading_text:
            references_section += markdown + "\n\n"
        else:
            all_markdown_sections.append(markdown)

    full_text = "\n\n".join(all_markdown_sections).strip()

    return {
        "title": title,
        "authors": authors,
        "text": full_text,
        "references_markdown": references_markdown.strip()
    }

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
# ARR-EMNLP Dataset loader
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
            "tei_data": {},
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
                logger.warning(f"⚠️  Skipping malformed meta.json in {paper_id}")
        else:
            logger.warning(f"⚠️  No usable meta.json in v1 for paper {paper_id}")

        # -------- paper.tei --------------------------------------------------
        tei_path = v1_dir / "paper.tei"
        if tei_path.exists() and tei_path.stat().st_size:
            try:
                paper_data["tei_data"] = tei_to_docklink_style_dict(tei_path)
                paper_data["tei_data"]["authors"] = paper_data.get("metadata", {}).get("authors", [])
            except json.JSONDecodeError:
                logger.warning(f"⚠️  Skipping malformed docling JSON for {paper_id}")

        # -------- reviews.json ----------------------------------------------
        reviews_path = v1_dir / "reviews.json"
        if reviews_path.exists() and reviews_path.stat().st_size:
            try:
                paper_data["reviews"] = format_reviews(
                    json.loads(reviews_path.read_text())
                )
            except json.JSONDecodeError:
                logger.warning(f"⚠️  Skipping malformed reviews.json for {paper_id}")

        # ❗ Skip if no reviews are present
        if not paper_data["reviews"]:
            logger.warning(f"⚠️  No reviews found for paper {paper_id}, skipping entry.")
            continue

        # -------- Extract reference titles with LLM (only for RAG evaluation) --------
        if rag_eval:
            reference_block = paper_data["tei_data"].get("references_markdown", "")
            if reference_block:
                titles = extract_titles_with_llm(reference_block, model=llm)
                paper_data["tei_data"]["reference_titles"] = titles

        dataset.append(paper_data)

    return dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_reviews(reviews_data):
    formatted_reviews = []
    for review in reviews_data:
        r = review.get("report", {})
        scores = review.get("scores", {})
        meta = review.get("meta", {})

        formatted_review = {
            "reviewer_id": review.get("rid", "[unknown]"),
            "paper_summary": r.get("paper_summary", ""),
            "summary_of_strengths": r.get("summary_of_strengths", ""),
            "summary_of_weaknesses": r.get("summary_of_weaknesses", ""),
            "comments_suggestions_and_typos": r.get("comments_suggestions_and_typos", ""),
            "ethical_concerns": r.get("ethical_concerns", ""),
            "scores": {
                "soundness": scores.get("soundness", None),
                "overall_assessment": scores.get("overall_assessment", None),
                "datasets": scores.get("datasets", None),
                "software": scores.get("software", None),
                "best_paper": scores.get("best_paper", "")
            },
            "reviewer_confidence": meta.get("confidence", None)
        }

        formatted_reviews.append(formatted_review)
    return formatted_reviews

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


# if __name__ == "__main__":
#     LOCAL TESTING ONLY: The following block is for manual/local testing and should not be run in production or on import.
#     main()