from nemoguardrails.actions import action

@action
def retrieve_documents(query: str) -> list:
    """
    Enterprise RAG retrieval.
    Must return a list of verified documents or an empty list.
    """
    # Call Pinecone / vector DB here
    return []

@action
def generate_grounded_answer(documents: list) -> str:
    """
    Generate answer strictly from provided documents.
    """
    return "Grounded answer based on documents."
