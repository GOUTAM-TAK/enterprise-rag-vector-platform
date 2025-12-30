from pathlib import Path
from typing import List, Dict
import asyncio
from pypdf import PdfReader
from docx import Document


class DocumentChunk:
    def __init__(self, text: str, metadata: Dict):
        self.text = text
        self.metadata = metadata


class AsyncDocumentLoader:
    """
    Async-safe document loader.
    CPU-heavy parsing is offloaded from event loop.
    """

    async def load(self, file_path: Path) -> List[DocumentChunk]:
        if file_path.suffix.lower() == ".pdf":
            return await asyncio.to_thread(self._load_pdf, file_path)
        elif file_path.suffix.lower() == ".docx":
            return await asyncio.to_thread(self._load_docx, file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _load_pdf(self, file_path: Path) -> List[DocumentChunk]:
        reader = PdfReader(str(file_path))
        chunks: List[DocumentChunk] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(
                    DocumentChunk(
                        text=text,
                        metadata={
                            "source": file_path.name,
                            "page": page_number,
                            "type": "pdf",
                        },
                    )
                )
        return chunks

    def _load_docx(self, file_path: Path) -> List[DocumentChunk]:
        doc = Document(str(file_path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            return []

        return [
            DocumentChunk(
                text="\n".join(paragraphs),
                metadata={
                    "source": file_path.name,
                    "type": "docx",
                },
            )
        ]
