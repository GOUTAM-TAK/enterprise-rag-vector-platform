def build_prompt(query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)

    return f"""
You are an AI assistant.

Context:
{context_block}

Question:
{query}

Answer:
"""
