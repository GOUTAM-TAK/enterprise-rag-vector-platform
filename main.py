from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.router import api_router
from app.rag_core.vectorstore.pinecone_client import PineconeClient
from app.rag_core.embeddings.embedder import AsyncSentenceEmbedder
from app.rag_core.llm.llm_registry import LLMRegistry
from app.core.config import settings
from app.core.logger import get_logger
from prometheus_client import make_asgi_app

logger = get_logger("startup")

metrics_app = make_asgi_app()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup initiated")

    # -------------------------
    # Initialize Pinecone (ONCE)
    # -------------------------
    pinecone_client = PineconeClient()
    pinecone_client.initialize()

    # -------------------------
    # Initialize Embedder (ONCE)
    # -------------------------
    embedder = AsyncSentenceEmbedder(
        model_name=settings.EMBEDDING_MODEL
    )

     # LLM Registry
    llm_registry = LLMRegistry()
    llm_registry.initialize()
    # -------------------------
    # Store in app.state
    # -------------------------
    app.state.pinecone = pinecone_client
    app.state.embedder = embedder
    app.state.llms = llm_registry

    logger.info("Shared resources initialized")
    logger.info("Application startup completed")

    yield  # ---- App is running ----

    logger.info("Application shutdown initiated")
    logger.info("Application shutdown completed")


app = FastAPI(
    title="Enterprise RAG Platform",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")
app.mount("/metrics", metrics_app)
