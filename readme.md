# Academic Peer Review Generation & Evaluation Pipeline

This repository provides a modular framework for generating, retrieving, and evaluating academic peer reviews using large language models (LLMs) and retrieval-augmented generation (RAG). The system supports both LLM-only and RAG-based review generation, and includes robust evaluation pipelines using both traditional metrics and LLM-based ELO judging.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Codebase Structure](#codebase-structure)
- [Main Pipelines](#main-pipelines)
- [Evaluation](#evaluation)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

This project enables:
- **Automated generation of academic peer reviews** using LLMs, with or without retrieval-augmented context.
- **Flexible retrieval** of relevant context chunks from a vector database (ChromaDB) to support RAG workflows.
- **Comprehensive evaluation** of generated reviews using both traditional metrics (BLANC, BLEURT) and LLM-based ELO judging.

The system is designed for research on LLM-based peer review, benchmarking, and ablation studies.

---

## Codebase Structure

```
Seminar/
├── chroma_db/                # ChromaDB utilities for vector storage and retrieval
│   ├── chroma.py
│   ├── chroma_utils.py
├── data_loader/              # Data loading, parsing, and conversion utilities
│   ├── arxiv_loader.py
│   ├── openalex_loader.py
│   ├── dataset_loader.py
│   ├── utils.py
├── evaluation/               # Evaluation pipelines and metrics
│   ├── run_evaluation.py
│   ├── util.py
│   ├── loader.py
│   ├── llm_judge/            # LLM-based ELO judging
│   ├── traditional_metrics/  # BLANC, BLEURT scoring
├── llm_only_pipeline/        # LLM-only review generation pipeline
│   ├── run_llm_only.py
│   ├── prompt_builder.py
├── models/                   # Model interfaces and OpenAI wrappers
│   ├── base.py
│   ├── generator.py
│   ├── openai_models.py
├── rag_pipeline/             # Retrieval-augmented generation pipeline
│   ├── run_rag.py
│   ├── prompt_builder.py
│   ├── retriever.py
├── requirements.txt          # Python dependencies
├── readme.md                 # This file
```

---

## Main Pipelines

### 1. **LLM-Only Pipeline** (`llm_only_pipeline/`)
- Generates peer reviews using an LLM without retrieval-augmented context.
- Loads papers, builds prompts, and saves generated reviews.
- Entry point: `run_llm_only.py`

### 2. **RAG Pipeline** (`rag_pipeline/`)
- Retrieves relevant context chunks from ChromaDB for each paper.
- Builds prompts with injected context and generates reviews using an LLM.
- Entry point: `run_rag.py`

### 3. **ChromaDB Utilities** (`chroma_db/`)
- Handles vector storage, retrieval, and collection management for context chunks.
- Used by the RAG pipeline for similarity search.

### 4. **Data Loading** (`data_loader/`)
- Loads and parses academic papers from various formats (arXiv, OpenAlex, TEI XML, Docling JSON).
- Prepares datasets for review generation and evaluation.

---

## Evaluation

### 1. **Traditional Metrics** (`evaluation/traditional_metrics/`)
- **BLANC**: Measures factual consistency between generated and reference reviews.
- **BLEURT**: Evaluates semantic similarity using pretrained models.

### 2. **LLM-Based ELO Judging** (`evaluation/llm_judge/`)
- Runs ELO-based tournaments where an LLM judges review pairs (human, LLM-only, RAG).
- Produces relative quality rankings for each system.

### 3. **Evaluation Runner** (`evaluation/run_evaluation.py`)
- Orchestrates the evaluation process, merges outputs, and saves results.

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd VanillaLLMvsRAG
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API keys:**
   - For OpenAI models, set your API key as an environment variable:
   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

---

## Usage

### 1. **Run LLM-Only Pipeline**
   ```bash
   python llm_only_pipeline/run_llm_only.py
   ```

### 2. **Run RAG Pipeline**
   ```bash
   python rag_pipeline/run_rag.py
   ```

### 3. **Run Evaluation**
   ```bash
   python evaluation/run_evaluation.py
   ```

- Outputs and intermediate results are saved in the respective pipeline directories.
- See code comments and docstrings for further customization and details.

---
