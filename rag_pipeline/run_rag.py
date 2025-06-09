import logging
from models.openai_models import OpenAIEmbeddingModel, OpenAILLM
from rag_pipeline.retriever import retrieve_context
from rag_pipeline.prompt_builder import build_prompt
from rag_pipeline.prompt_templates import RAG_TEMPLATE_V1
from rag_pipeline.generator import generate_answer
from chroma_db.chroma_utils import load_collection 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # Step 0: Init models
    embedder = OpenAIEmbeddingModel()
    llm = OpenAILLM()

    # Step 1: Load vector store collection
    collection = load_collection("papers")

    # Step 2: Define user query
    user_query = "What are the advantages of using retrieval augmented generation?"

    # Step 3: Embed query
    query_embedding = embedder.embed_text(user_query)

    # Step 4: Retrieve context from Chroma
    context_chunks = retrieve_context(collection, query_embedding, k=3)

    # Step 5: Build LLM prompt
    messages = build_prompt(user_query, context_chunks, template=RAG_TEMPLATE_V1)

    # Step 6: Generate answer from LLM
    answer = generate_answer(llm, messages)

    # Step 7: Output results
    print("\n🔍 User Query:", user_query)
    print("🗂 Retrieved Context Chunks:")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n--- Context Chunk {i} ---\n{chunk}\n")

    print("\n🧠 Answer:\n", answer)

if __name__ == "__main__":
    main()