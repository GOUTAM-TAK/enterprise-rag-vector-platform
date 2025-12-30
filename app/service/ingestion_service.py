import uuid
from pathlib import Path

from app.rag_core.ingestion.loader import AsyncDocumentLoader
from app.rag_core.ingestion.chunker import AsyncSentenceChunker
from app.rag_core.embeddings.embedder import AsyncSentenceEmbedder
from app.rag_core.vectorstore.pinecone_client import PineconeClient
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    """
    Coordinates document ingestion pipeline.
    """

    @staticmethod
    async def ingest_document(file_path: Path):
        document_id = str(uuid.uuid4())

        try:
            logger.info(f"Starting ingestion | doc_id={document_id}")

            # ---------- Load document ----------
            loader = AsyncDocumentLoader()
            raw_documents = await loader.load(file_path)

            if not raw_documents:
                logger.warning(f"No content extracted | doc_id={document_id}")
                return

            # ---------- Chunk document ----------
            chunker = AsyncSentenceChunker(
                model_name=settings.EMBEDDING_MODEL,
                max_tokens=settings.CHUNK_SIZE,
                overlap_tokens=settings.CHUNK_OVERLAP,
            )
            chunks = await chunker.split(raw_documents)

            logger.info(
                f"Chunking completed | doc_id={document_id} | chunks={len(chunks)}"
            )

            # ---------- Embed chunks ----------
            embedder = AsyncSentenceEmbedder(
                model_name=settings.EMBEDDING_MODEL
            )

            texts = [chunk.text for chunk in chunks]
            embeddings = await embedder.embed_texts(texts)

            # ---------- Prepare Pinecone vectors ----------
            vectors = []
            for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                vectors.append(
                    {
                        "id": f"{document_id}-{i}",
                        "values": vector,
                        "metadata": {
                            **chunk.metadata,
                            "document_id": document_id,
                        },
                    }
                )

            # ---------- Upsert to Pinecone ----------
            pinecone = PineconeClient()
            await pinecone.upsert(
                vectors=vectors,
                namespace=document_id,
            )

            logger.info(
                f"Ingestion completed | doc_id={document_id} | vectors={len(vectors)}"
            )

        except Exception:
            logger.exception(f"Ingestion failed | doc_id={document_id}")
