import uuid
from pathlib import Path
import aiofiles
from fastapi import UploadFile

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class FileUtils:
    @staticmethod
    async def save_upload_async(file: UploadFile) -> Path:
        filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = UPLOAD_DIR / filename

        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        return file_path
