"""
Utility functions for evaluation dataset integrity and filtering.

This module provides helpers for filtering complete entries, checking dataset integrity,
and logging missing or incomplete review fields for evaluation workflows.
"""
import logging

logger = logging.getLogger(__name__)

def filter_complete_entries(merged_data: list[dict]) -> list[dict]:
    """
    Filters the merged dataset to only include entries where:
    - LLM-only review is present
    - LLM+RAG review is present
    - Human review is present
    """

    required_fields = ["reviews", "llm_generated_review", "llm_plus_rag_generated_review"]
    filtered = []
    dropped = 0

    for entry in merged_data:
        missing = [field for field in required_fields if not entry.get(field)]
        if missing:
            dropped += 1
            paper_id = entry.get("paper_id", "[unknown id]")
            logger.warning(f"⚠️ Dropping paper {paper_id} due to missing fields: {', '.join(missing)}")
            continue

        filtered.append(entry)

    logger.info(f"✅ Retained {len(filtered)} entries with complete data. Dropped {dropped} incomplete entries.")
    return filtered


# evaluation/util.py  (add this helper)

from collections import defaultdict
import logging
from typing import Iterable, List, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyse_dataset_integrity(
    records: Iterable[Dict[str, Any]],
    *,
    arr_emnlp_root: Path,
    id_key: str = "paper_id",
) -> None:
    """
    Check required fields + report missing/extra paper IDs (all treated as *strings*).
    """

    # 1) Identify which generated-review field exists
    gen_keys = ["llm_generated_review", "llm_plus_rag_generated_review"]
    present  = [k for k in gen_keys if any(k in r for r in records)]
    if len(present) != 1:
        raise ValueError(f"Need exactly one of {gen_keys}; found {present or 'none'}")
    gen_key  = present[0]
    required = [gen_key, "reviews"]

    # 2) Scan dataset
    missing_by_field, missing_by_paper = defaultdict(list), defaultdict(list)
    present_ids: List[str] = []

    for idx, rec in enumerate(records):
        pid = str(rec.get(id_key, idx))      # keep every ID as *string*
        present_ids.append(pid)

        for fld in required:
            if rec.get(fld) in (None, "", []):
                missing_by_field[fld].append(pid)
                missing_by_paper[pid].append(fld)

    # 3) Expected IDs = folder names under data/ARR-EMNLP
    expected_ids = sorted(
        p.name for p in arr_emnlp_root.iterdir() 
        if p.is_dir() and p.name.isdigit()
    )
    missing_ids    = _sort_ids_numerically(set(expected_ids) - set(present_ids))
    unexpected_ids = _sort_ids_numerically(set(present_ids)  - set(expected_ids))

    # 4) Structured log output
    logger.info("📊 DATASET INTEGRITY CHECK — generated field: %s", gen_key)
    total_in_dataset = len(present_ids)         

    for fld in required:
        ids = sorted(missing_by_field[fld])
        logger.warning(
            "⚠️  Field '%s' missing/empty for %d/%d entries: %s",
            fld, len(ids), total_in_dataset, ids
        )

    if missing_ids:
        logger.warning("⚠️  Missing paper_ids (folder exists, absent in dataset): %s",
                       missing_ids)
    if unexpected_ids:
        logger.warning("⚠️  Extra paper_ids (in dataset, no matching folder): %s",
                       unexpected_ids)

    ok_entries = len(present_ids) - len(missing_by_paper)
    logger.info("✅ %d/%d dataset entries have all required fields.",
                ok_entries, len(present_ids))

    logger.info("📦 Dataset contains %d of %d expected papers.",
                len(present_ids), len(expected_ids))
    

def analyse_joint_dataset_integrity(
    records: Iterable[Dict[str, Any]],
    id_key: str = "paper_id",
) -> None:
    """
    Check that each entry in the merged dataset has all three review fields.
    Logs missing fields per paper_id.
    """
    required_fields = ["reviews", "llm_generated_review", "llm_plus_rag_generated_review"]
    missing_by_field = {field: [] for field in required_fields}
    missing_by_paper = {}

    present_ids: List[str] = []

    for idx, rec in enumerate(records):
        pid = str(rec.get(id_key, idx))
        present_ids.append(pid)
        missing = [field for field in required_fields if rec.get(field) in (None, "", [])]
        if missing:
            missing_by_paper[pid] = missing
            for field in missing:
                missing_by_field[field].append(pid)

    total = len(present_ids)
    complete = total - len(missing_by_paper)

    logger.info("📊 JOINT DATASET INTEGRITY CHECK")
    for field in required_fields:
        ids = sorted(missing_by_field[field])
        logger.warning(
            "⚠️  Field '%s' missing/empty for %d/%d entries: %s",
            field, len(ids), total, ids
        )
    if missing_by_paper:
        logger.warning("⚠️  Papers with missing fields: %s", missing_by_paper)
    logger.info("✅ %d/%d entries have all required review fields.", complete, total)


def _sort_ids_numerically(id_list):
    """Return IDs sorted as integers whenever they look like integers."""
    return sorted(id_list, key=lambda x: int(x) if x.isdigit() else x)