from app.rag_core.prompt.prompt_builder import build_prompt

class RAGChain:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def stream(self, query: str, contexts: list[str]):
        prompt = build_prompt(query, contexts)

        async for token in self.llm.stream(prompt):
            yield token
