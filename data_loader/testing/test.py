"""
Full ARR-EMNLP reference-pipeline smoke-test.

Run from repo root with:
    python -m data_loader.testing.test [--dataset ./data/ARR-EMNLP] [--verbose]

Steps
-----
1.  Load the ARR dataset        → canonical   paper-dict list
2.  Extract reference sections  → list[str]   (unique paper titles)
3.  Fetch OpenAlex metadata     → list[dict]  (title, abstract, …)
4.  Embed + persist in Chroma   → side effect (vector-db)

Nothing is written to disk except via `store_openalex_papers_in_vector_db`.
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from models.openai_models import OpenAILLM 

import logging

# ------------------------------------------------------------------------------
# local imports (add project root on PYTHONPATH so "python -m" works everywhere)
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from data_loader.dataset_loader import load_arr_emnlp_dataset
from data_loader.utils import extract_titles_with_llm
from data_loader.openalex_loader import (
    fetch_abstracts_for_titles,
    store_openalex_papers_in_vector_db,
)

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
logger = logging.getLogger("arr-pipeline")

def load_dataset(path: Path) -> list[dict]:
    papers = load_arr_emnlp_dataset(path)
    if not papers:
        raise RuntimeError(f"Dataset @ {path} could not be loaded or is empty.")
    logger.info("✅  Loaded %d ARR-EMNLP papers", len(papers))
    return papers


def collect_reference_titles(papers: list[dict], verbose: bool = False) -> list[str]:
    titles: set[str] = set()

    for p in papers:
        sections = p.get("docling_paper", {}).get("sections", [])
        if verbose:
            logger.debug("📄  %s - %d sections", p["paper_id"], len(sections))
        for t in extract_titles_with_llm(sections):
            titles.add(t)

    if not titles:
        raise RuntimeError("No reference titles found in any paper.")
    logger.info("🔎  Extracted %d unique reference titles", len(titles))
    return sorted(titles)


def fetch_and_store(titles: list[str]) -> None:
    logger.info("🌐  Querying OpenAlex for %d titles …", len(titles))
    papers = fetch_abstracts_for_titles(titles)

    if not papers:
        raise RuntimeError("OpenAlex returned 0 results – aborting.")
    logger.info("📄  Retrieved %d papers with abstracts", len(papers))

    logger.info("💾  Persisting embeddings to Chroma …")
    store_openalex_papers_in_vector_db(papers)
    logger.info("✅  Pipeline completed – abstracts embedded and stored.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ARR reference-pipeline smoke-test")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "ARR-EMNLP",
        help="Folder containing the ARR-EMNLP corpus (default: %(default)s)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging for section dumps",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname).1s %(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        # Step 1: Load full dataset (including reference_titles already populated)
        llm = OpenAILLM(model_name="gpt-4o")
        papers = load_arr_emnlp_dataset(args.dataset, llm=llm)

        # Step 2: Collect all unique reference titles from the dataset
        all_titles = {
            t.strip()
            for p in papers
            for t in p.get("docling_paper", {}).get("reference_titles", [])
        }

        if not all_titles:
            raise RuntimeError("No reference titles found in any paper.")
        logger.info("🔎  Extracted %d unique reference titles", len(all_titles))

        if args.verbose:
            for t in sorted(all_titles):
                logger.debug(f"→ {t}")

        # Step 3 + 4: Fetch OpenAlex metadata and persist to ChromaDB
        fetch_and_store(sorted(all_titles))

    except Exception as exc:
        logger.exception("❌  Pipeline failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()