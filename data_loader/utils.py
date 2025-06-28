import re
import json
import tempfile
from pathlib import Path
from models.openai_models import OpenAILLM
import os
import logging

logger = logging.getLogger(__name__)


def load_existing_outputs(path: str) -> dict:
    if os.path.exists(path):
        if os.path.getsize(path) == 0:
            print(f"⚠️  Existing output file at {path} is empty. Returning empty dict.")
            return {}
        with open(path, "r") as f:
            return {entry["paper_id"]: entry for entry in json.load(f)}
    return {}


def split_markdown_sections(md: str) -> list[dict]:
    """Split markdown into sections based on ATX headings (one or more #).

    Returns a list like [{"heading": "Intro", "text": "…"}, …].
    """
    sections: list[dict] = []
    current = {"heading": None, "text": ""}

    for line in md.splitlines():
        if re.match(r"^#{1,6} ", line):
            # Start a new section
            if current["heading"] or current["text"].strip():
                sections.append(current)
            current = {"heading": line.lstrip("# ").strip(), "text": ""}
        else:
            current["text"] += line + "\n"

    if current["heading"] or current["text"].strip():
        sections.append(current)
    return sections


def get_title_and_authors_from_furniture(docling_json: dict) -> tuple[str, list[str]]:
    """Pull best‑guess title & author list from the docling "furniture" block."""
    title = "[no title]"
    authors: list[str] = []

    furniture = docling_json.get("furniture", {})
    for child in furniture.get("children", []):
        ref = child.get("$ref") or child.get("ix")
        node = (
            docling_json.get("texts", {},).get(ref, {})
            or docling_json.get("nodes", {},).get(ref, {})
        )
        ntype = node.get("ntype") or node.get("label")

        if ntype == "title":
            title = node.get("content") or node.get("text") or title
        elif ntype == "authors":
            raw = node.get("content") or node.get("text", "")
            authors = [a.strip() for a in re.split(r"[;,]", raw) if a.strip()]

    return title, authors


def extract_titles_with_llm(reference_block: str, model: OpenAILLM) -> list[str]:
    prompt = f"""
    You are a scholarly assistant. Extract **only** the paper titles from the following references.

    Ignore authors, venues, or years — return just a clean **JSON list** of paper titles, like:
    ["Title One", "Another Title", "Final Title"]

    References:
    {reference_block}
    """
    try:
        response_text = model.generate_text([
            {"role": "system", "content": "You extract paper titles from scientific references."},
            {"role": "user", "content": prompt}
        ])

        # DEBUG: See exactly what the model returns
        logger.debug(f"🔍 Raw LLM output:\n{response_text}")

        # Extract just the JSON list from the text
        match = re.search(r"\[\s*\".*?\"\s*(?:,\s*\".*?\"\s*)*\]", response_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in model response")

        titles = json.loads(match.group(0))
        return [t.strip().rstrip(".") for t in titles if isinstance(t, str)]

    except Exception as e:
        logger.error(f"❌ Failed to extract titles with LLM: {e}")
        return []
    

import json
import re

def parse_review_json(response, paper_id):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning(f"🔧 Attempting to fix malformed JSON for paper {paper_id}...")

        # Remove markdown formatting if present
        response = response.strip().removeprefix("```json").removesuffix("```").strip()

        # Try to fix single quotes to double quotes (risky, but can work in controlled prompts)
        response = re.sub(r"(?<!\\)'", '"', response)

        # Try parsing again
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"❌ Final JSON parse failed for paper {paper_id}: {e}")
            return None