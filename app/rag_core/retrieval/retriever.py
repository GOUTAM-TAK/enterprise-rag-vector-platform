class Retriever:
    def __init__(self, pinecone_client):
        self.pinecone = pinecone_client

    async def retrieve(self, vector, namespace, top_k=5):
        return await self.pinecone.query(
            vector=vector,
            namespace=namespace,
            top_k=top_k,
        )
