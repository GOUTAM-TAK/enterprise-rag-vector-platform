from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import asyncio
from pathlib import Path

from app.service.ingestion_service import IngestionService
from app.core.config import settings
from app.utils.file_utils import FileUtils
from app.utils.rag_utils import RAGUtils

router = APIRouter()


@router.post("/pdf")
async def ingest_pdf(request: Request,file: UploadFile = File(...),rag_access_level:str = "public"):
    # ---------- Validation ----------
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    level, access_rank = RAGUtils.validate_rag_access_level(rag_access_level)

    # ---------- Save file ----------
    file_path: Path = await FileUtils.save_upload_async(file=file)

    # ---------- Fire-and-forget async ingestion ----------
    asyncio.create_task(
        IngestionService.ingest_document(file_path=file_path,
                                         request=request, 
                                         rag_access_level=level,
                                         access_rank=access_rank)
                                )

    return {
        "status": "INGESTION_STARTED",
        "filename": file.filename,
        "message": "PDF ingestion started asynchronously",
    }
