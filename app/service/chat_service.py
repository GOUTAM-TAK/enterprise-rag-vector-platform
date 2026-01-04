from fastapi import WebSocket
from time import perf_counter
from app.core.config import settings
from app.core.logger import get_logger
from app.rag_core.chain.rag_chain import RAGChain
from app.core.metrics import (
    CHAT_REQUESTS_TOTAL,
    CHAT_ERRORS_TOTAL,
    EMBEDDING_LATENCY,
    RETRIEVAL_LATENCY,
    RETRIEVED_CONTEXTS,
    CHAT_TOTAL_LATENCY,
    LLM_FIRST_TOKEN_LATENCY,
)
from app.utils.rag_utils import RAGUtils

logger = get_logger(__name__)


class ChatService:
    """
    High-performance RAG chat service with Prometheus metrics.
    """

    @staticmethod
    async def handle_chat(
        ws: WebSocket,
        payload: dict,
    ):
        start_time = perf_counter()

        query = payload.get("query")
        namespace = payload.get("namespace")
        model_name = payload.get("model")

        if not query:
            CHAT_ERRORS_TOTAL.inc()
            await ws.send_json({
                "event_type": "error",
                "message": "Query is required",
            })
            return

        # -------------------------
        # Metrics: request count
        # -------------------------
        CHAT_REQUESTS_TOTAL.labels(model=model_name).inc()

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

        try:
            # -------------------------
            # 1. Embed query
            # -------------------------
            with EMBEDDING_LATENCY.time():
                query_vector = await embedder.embed_query(query)

            # -------------------------
            # 2. Vector retrieval (Pinecone)
            # -------------------------
            level, access_rank = RAGUtils.validate_rag_access_level(
                rag_access_level=payload.get("rag_access_level","public"),
                raise_http=False,
            )

            with RETRIEVAL_LATENCY.time():
                result = await pinecone.query(
                    vector=query_vector,
                    namespace=settings.NAME_SPACE,
                    top_k=5,
                    include_metadata=True,
                    metadata_filter={
                        "rag_access_level_rank": {"$lte": access_rank}
                    },
                )

            matches = result.get("matches", [])

            logger.info(
                "Vector retrieval completed | namespace=%s | matches=%d",
                namespace,
                len(matches),
            )

            # -------------------------
            # 2.1 Extract usable contexts
            # -------------------------
            contexts = [
                match["metadata"]["text"]
                for match in matches
                if "text" in match.get("metadata", {})
            ]

            RETRIEVED_CONTEXTS.observe(len(contexts))

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
                CHAT_ERRORS_TOTAL.inc()
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

            first_token = True
            llm_start = perf_counter()

            async for token in rag_chain.stream(query, contexts):
                if first_token:
                    LLM_FIRST_TOKEN_LATENCY.observe(
                        perf_counter() - llm_start
                    )
                    first_token = False

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

        except Exception as exc:
            CHAT_ERRORS_TOTAL.inc()
            logger.exception(
                "Chat processing failed | namespace=%s | model=%s",
                namespace,
                model_name,
            )
            await ws.send_json({
                "event_type": "error",
                "message": "Internal server error",
            })
            raise exc

        finally:
            CHAT_TOTAL_LATENCY.observe(
                perf_counter() - start_time
            )
