from fastapi import APIRouter
from app.api.ingestion import router as ingestion_router
# Feature routers

# Future routers (placeholders)
# from app.api.retrieval import router as retrieval_router
# from app.api.evaluation import router as evaluation_router

api_router = APIRouter()

# -------------------------
# Health / Meta
# -------------------------
@api_router.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "UP",
        "service": "enterprise-rag-platform",
    }

# -------------------------
# RAG Ingestion APIs
# -------------------------
api_router.include_router(
    ingestion_router,
    prefix="/ingest",
    tags=["RAG Ingestion"],
)

# -------------------------
# Future Expansion
# -------------------------
# api_router.include_router(
#     retrieval_router,
#     prefix="/retrieve",
#     tags=["RAG Retrieval"],
# )
#
# api_router.include_router(
#     evaluation_router,
#     prefix="/evaluate",
#     tags=["RAG Evaluation"],
# )
