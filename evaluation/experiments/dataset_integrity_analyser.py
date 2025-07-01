from pathlib import Path
import logging
from evaluation.util import analyse_dataset_integrity, analyse_joint_dataset_integrity
from evaluation.loader import load_json

logger = logging.getLogger(__name__)

def analyse_dataset_integrity(dataset):
    rag_path = ""
    llm_only_path = ""
    #rag_path = "rag_pipeline/output.json"
    #llm_only_path = "llm_only_pipeline/output.json"
    eval_path = "evaluation/dataset_with_traditional_scores.json"

    if eval_path:
        logger.info(f"✅ Merged dataset analysis before running evaluation metrics from {eval_path}")
        analyse_joint_dataset_integrity(
            dataset        
        )
    else: 
        logger.info(f"✅Evaluation integrity of both pipeline outputs {rag_path} and {llm_only_path} before merging")
        data = load_json(rag_path)      
        analyse_dataset_integrity(
            data,
            arr_emnlp_root=Path("data/ARR-EMNLP"),         
        )

def main():

    rag_path = ""
    llm_only_path = ""
    #rag_path = "rag_pipeline/output.json"
    #llm_only_path = "llm_only_pipeline/output.json"
    eval_path = "evaluation/dataset_with_traditional_scores.json"

    if eval_path:
        logger.info(f"✅ Evaluation joint evaluation dataset before running evaluation metrics from {eval_path}")
        data = load_json(eval_path)      
        analyse_joint_dataset_integrity(
            data         
        )
    else: 
        logger.info(f"✅Evaluation integrity of both pipeline outputs {rag_path} and {llm_only_path} before merging")
        data = load_json(rag_path)      
        analyse_dataset_integrity(
            data,
            arr_emnlp_root=Path("data/ARR-EMNLP"),         
        )

if __name__ == "__main__":
    main()