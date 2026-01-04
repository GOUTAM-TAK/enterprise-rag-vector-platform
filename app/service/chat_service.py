from fastapi import WebSocket
from time import perf_counter

from app.core.logger import get_logger
from app.rag_core.chain.rag_chain import RAGChain

logger = get_logger(__name__)


class ChatService:
    """
    High-performance RAG chat service.
    """

    @staticmethod
    async def handle_chat(
        ws: WebSocket,
        payload: dict,
    ):
        query = payload.get("query")
        namespace = payload.get("namespace")
        model_name = payload.get("model")

        if not query:
            await ws.send_json({
                "event_type": "error",
                "message": "Query is required",
            })
            return

        # -------------------------
        # Fetch shared resources
        # -------------------------
        embedder = ws.app.state.embedder
        pinecone = ws.app.state.pinecone
        llm_registry = ws.app.state.llms

        logger.info(
            "Chat request received | namespace=%s | model=%s",
            namespace,
            model_name,
        )

        # -------------------------
        # 1. Embed query
        # -------------------------
        embed_start = perf_counter()
        query_vector = await embedder.embed_query(query)
        embed_time = perf_counter() - embed_start

        logger.info(
            "Query embedding completed | latency=%.3fs",
            embed_time,
        )

        # -------------------------
        # 2. Vector retrieval (Pinecone)
        # -------------------------
        retrieval_start = perf_counter()

        result = await pinecone.query(
            vector=query_vector,
            namespace=namespace,
            top_k=5,
            include_metadata=True,
        )

        retrieval_time = perf_counter() - retrieval_start
        matches = result.get("matches", [])

        logger.info(
            "Vector retrieval completed | namespace=%s | matches=%d | latency=%.3fs",
            namespace,
            len(matches),
            retrieval_time,
        )

        # -------------------------
        # 2.1 Extract usable contexts
        # -------------------------
        contexts = [
            match["metadata"]["text"]
            for match in matches
            if "text" in match.get("metadata", {})
        ]

        logger.info(
            "Context extraction completed | usable_contexts=%d",
            len(contexts),
        )

        if not contexts:
            logger.warning(
                "No usable context found | namespace=%s | query=%s",
                namespace,
                query[:100],
            )

        # -------------------------
        # 3. Fetch preloaded LLM
        # -------------------------
        llm_client = llm_registry.get(model_name)

        if not llm_client:
            logger.error(
                "Requested model not found in registry | model=%s",
                model_name,
            )
            await ws.send_json({
                "event_type": "error",
                "message": "Requested model not available",
            })
            return

        # -------------------------
        # 4. Streaming RAG chain
        # -------------------------
        rag_chain = RAGChain(llm_client)

        logger.info(
            "Starting RAG streaming | model=%s | contexts=%d",
            model_name,
            len(contexts),
        )

        async for token in rag_chain.stream(query, contexts):
            await ws.send_json({
                "event_type": "chat_stream",
                "token": token,
            })

        await ws.send_json({
            "event_type": "chat_complete",
        })

        logger.info(
            "Chat completed | namespace=%s | model=%s",
            namespace,
            model_name,
        )
