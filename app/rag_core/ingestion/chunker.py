import asyncio
from typing import List
from app.rag_core.ingestion.loader import DocumentChunk
from app.rag_core.embeddings.tokenizer import SentenceTokenizerProvider


class AsyncSentenceChunker:
    """
    Async-safe chunker optimized for Sentence Transformer embeddings.

    - No model loading
    - Tokenizer-only dependency
    - Enterprise-safe design
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        max_tokens: int = 256,
        overlap_tokens: int = 40,
    ):
        self.tokenizer = SentenceTokenizerProvider.get_tokenizer(model_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    async def split(self, documents: List[DocumentChunk]) -> List[DocumentChunk]:
        return await asyncio.to_thread(self._split_sync, documents)

    def _split_sync(self, documents: List[DocumentChunk]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []

        for doc in documents:
            paragraphs = [p for p in doc.text.split("\n") if p.strip()]
            buffer_tokens = []
            buffer_text = ""

            for para in paragraphs:
                para_tokens = self.tokenizer.encode(para, add_special_tokens=False)

                if len(buffer_tokens) + len(para_tokens) <= self.max_tokens:
                    buffer_tokens.extend(para_tokens)
                    buffer_text += " " + para
                else:
                    if buffer_text.strip():
                        chunks.append(
                            DocumentChunk(
                                text=buffer_text.strip(),
                                metadata={**doc.metadata},
                            )
                        )

                    overlap = buffer_tokens[-self.overlap_tokens :]
                    buffer_tokens = overlap + para_tokens
                    buffer_text = self.tokenizer.decode(buffer_tokens)

            if buffer_text.strip():
                chunks.append(
                    DocumentChunk(
                        text=buffer_text.strip(),
                        metadata={**doc.metadata},
                    )
                )

        return chunks
