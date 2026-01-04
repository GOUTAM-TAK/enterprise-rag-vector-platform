class Retriever:
    def __init__(self, pinecone_client):
        self.pinecone = pinecone_client

    async def retrieve(self, vector, namespace, access_rank, top_k=5):
        return await self.pinecone.query(
            vector=vector,
            namespace=namespace,
            top_k=top_k,
            metadata_filter={
                "rag_access_level_rank": {"$lte": access_rank}
            },
        )
