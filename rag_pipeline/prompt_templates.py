# rag_pipeline/prompt_templates.py

# Example simple templates (you can make them callable if needed)

# Template with context injection
RAG_TEMPLATE_V1 = """You are a helpful assistant. Use only the following context to answer the question.
If you cannot answer, say: "I don't know."

Context:
{context}

Question: {question}
"""
