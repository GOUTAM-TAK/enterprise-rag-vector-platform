from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.router import api_router
from app.rag_core.vectorstore.pinecone_client import PineconeClient
from app.core.logger import get_logger

logger = get_logger("startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Executes startup and shutdown logic.
    """
    logger.info("Application startup initiated")

    pinecone_client = PineconeClient()
    pinecone_client.initialize()

    logger.info("Pinecone initialized successfully")
    logger.info("Application startup completed")

    yield  # ---- Application is running ----

    logger.info("Application shutdown initiated")
    # Place cleanup logic here if needed
    logger.info("Application shutdown completed")


app = FastAPI(
    title="Enterprise RAG Platform",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")
