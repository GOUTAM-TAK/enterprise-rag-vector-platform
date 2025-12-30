import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
import torch


class AsyncSentenceEmbedder:
    """
    Async-safe SentenceTransformer embedder.

    Design principles:
    - Async public interface
    - CPU/GPU-bound embedding runs outside event loop
    - Cost-efficient (local embeddings)
    - GPU automatically used if available
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        normalize_embeddings: bool = True,
    ):
        """
        :param model_name: SentenceTransformer model name
        :param device: 'cuda', 'cpu', or None (auto-detect)
        :param normalize_embeddings: cosine-similarity friendly vectors
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts (used during ingestion).
        """
        return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query (used during retrieval).
        """
        embeddings = await asyncio.to_thread(self._embed_sync, [query])
        return embeddings[0]

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous embedding logic (GPU/CPU-bound).
        Runs in a worker thread to protect event loop.
        """

        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Convert numpy arrays to plain Python lists
        return vectors.tolist()
