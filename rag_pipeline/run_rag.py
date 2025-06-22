import json
import re
from rag_pipeline.retriever import retrieve_context
from rag_pipeline.prompt_builder import build_prompt
from rag_pipeline.prompt_templates import RAG_TEMPLATE_V1
from chroma_db.chroma import initiate_chroma_db
from data_loader.dataset_loader import load_arr_emnlp_dataset
from data_loader.openalex_loader import fetch_abstracts_for_titles, store_openalex_papers_in_vector_db
from rag_pipeline.prompt_builder import build_review_prompt
from acl_review_guidelines import review_guidelines
import logging
from models.openai_models import OpenAIEmbeddingModel
from models.openai_models import OpenAILLM
from sample_papers import sample_paper_reviews

logger = logging.getLogger("openalex-loader")
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():

    # Step 0: Init models and ChromaDB collection name
    embedder = OpenAIEmbeddingModel()
    llm = OpenAILLM()
    collection_name = "papers"

    # Step 2: Load dataset
    dataset = load_arr_emnlp_dataset("./data/ARR-EMNLP", llm=llm, rag_eval=True)
    logger.info(f"✅ Loaded dataset with {len(dataset)} entries.")

    # Step 3: Retrieve papers for references and store in ChromaDB
    papers_to_be_embedded = []

    for paper in dataset:
        doc = paper["docling_paper"]

        papers = fetch_abstracts_for_titles(doc.get("reference_titles", []))
        if not papers:
            logger.warning(f"No papers found for references in {paper['paper_id']}.")
            continue

        logger.info(f"Found {len(papers)} papers for references in {paper['paper_id']}.")
        papers_to_be_embedded.extend(papers)
    
    logger.info(f"Total papers to be embedded: {len(papers_to_be_embedded)}")
    logger.info("Storing papers in ChromaDB...")

    store_openalex_papers_in_vector_db(papers_to_be_embedded, collection_name=collection_name)
    logger.info("Successfully stored papers in ChromaDB.")

    # Step 3.1: Initiate ChromaDB and get collection
    chroma = initiate_chroma_db("./chroma_db")
    collection = chroma.get_or_create_collection(collection_name)

    # Step 4: Generate LLM reviews with RAG
    for paper in dataset:
        doc = paper["docling_paper"]

        if not paper["metadata"].get("abstract"):
            logger.warning(f"Skipping paper {paper['paper_id']} without abstract.")
            continue

        # Construct user query (e.g. full abstract as query)
        query_text = f"Title: {paper['metadata'].get('title', '')}\nAbstract: {paper['metadata'].get('abstract', '')}"
        query_embedding = embedder.embed_text(query_text)

        # Retrieve context chunks from ChromaDB
        context_chunks = retrieve_context(collection, query_embedding, k=5)
        paper["retrieved_chunks"] = context_chunks

        # Build review prompt
        prompt = build_review_prompt(
            paper={
                "title": paper["metadata"].get("title", "[no title]"),
                "abstract": paper["metadata"].get("abstract", ""),
                "full_text": doc.get("full_text"),
            },
            context_chunks=context_chunks,
            guidelines=review_guidelines,
            sample_reviews=sample_paper_reviews
        )

        # print(f"🔍 Generated prompt for paper {paper['paper_id']}:\n{prompt}\n")
        
        # Generate and parse review
        try:
            raw_review = llm.generate_text([
                {"role": "system", "content": "You are an academic peer reviewer."},
                {"role": "user", "content": prompt}
            ])

            if raw_review.strip().startswith("```json"):
                raw_review = re.sub(r"^```json\s*", "", raw_review.strip())
                raw_review = re.sub(r"\s*```$", "", raw_review.strip())

            parsed_review = json.loads(raw_review)
            paper["llm_plus_rag_generated_review"] = parsed_review

        except Exception as e:
            logger.error(f"Failed to generate review for {paper['paper_id']}: {e}")
            paper["llm_generated_review"] = None

    # Step 4: Pretty print results
    for paper in dataset:
        title = paper["metadata"].get("title", "[No Title]")
        human_reviews = paper.get("reviews", [])
        llm_review = paper.get("llm_plus_rag_generated_review")
        retrieved_chunks = paper["retrieved_chunks"]

        print("\n" + "=" * 100)
        print(f"📄 Title: {title}")
        print("-" * 100)

        # Human Review
        if human_reviews:
            print("🧑 Human Review (first):")
            print(json.dumps(human_reviews[0], indent=2))
        else:
            print("🧑 Human Review: Not available")

        # LLM-plus-Rag Generated Review
        print("\n🤖 LLM-plus-RAG Generated Review (with RAG):")
        if llm_review:
            print(json.dumps(llm_review, indent=2))
        else:
            print("❌ Failed to generate LLM review.")

        # Retrieved Context
        print("\n📚 Retrieved Context Chunks:")
        if retrieved_chunks:
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"\n--- Chunk {i} ---\n{chunk}")
        else:
            print("❌ No context retrieved.")

        print("=" * 100)


if __name__ == "__main__":
    main()