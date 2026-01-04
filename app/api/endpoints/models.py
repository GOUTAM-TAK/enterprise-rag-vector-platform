from fastapi import APIRouter, UploadFile, File, HTTPException, Request


from app.core.config import settings


router = APIRouter()

@router.get("/models")
async def list_models(request: Request):
    return {
        "models": request.app.state.llms.list_models(),
        "default": settings.NVIDIA_DEFAULT_MODEL
    }
