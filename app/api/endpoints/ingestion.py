from fastapi import APIRouter, UploadFile, File, HTTPException
import asyncio
from pathlib import Path

from app.service.ingestion_service import IngestionService
from app.core.config import settings
from app.utils.file_utils import FileUtils

router = APIRouter()


@router.post("/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    # ---------- Validation ----------
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # ---------- Save file ----------
    file_path: Path = await FileUtils.save_upload_async(file=file)

    # ---------- Fire-and-forget async ingestion ----------
    asyncio.create_task(
        IngestionService.ingest_document(file_path)
    )

    return {
        "status": "INGESTION_STARTED",
        "filename": file.filename,
        "message": "PDF ingestion started asynchronously",
    }
