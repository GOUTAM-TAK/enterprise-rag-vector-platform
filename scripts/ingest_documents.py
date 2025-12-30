import asyncio
from pathlib import Path
from rag_core.ingestion.loader import AsyncDocumentLoader
from rag_core.ingestion.chunker import AsyncSentenceChunker
from src.rag_core.embeddings.embedder import AsyncSentenceEmbedder

DATA_DIR = Path("data/raw")


async def main():
    loader = AsyncDocumentLoader()
    chunker = AsyncSentenceChunker(
        model_name="all-MiniLM-L6-v2",
        max_tokens=256,
        overlap_tokens=40,
    )
    embedder = AsyncSentenceEmbedder(
        model_name="all-MiniLM-L6-v2"
    )

    all_chunks = []

    for file_path in DATA_DIR.iterdir():
        if not file_path.is_file():
            continue

        print(f"Ingesting: {file_path.name}")

        docs = await loader.load(file_path)
        chunks = await chunker.split(docs)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks created: {len(all_chunks)}")

    # Embed a sample to validate pipeline
    sample_texts = [chunk.text for chunk in all_chunks[:5]]
    embeddings = await embedder.embed_texts(sample_texts)

    print(f"Sample embedding vector size: {len(embeddings[0])}")
    print("Ingestion pipeline validated successfully.")


if __name__ == "__main__":
    asyncio.run(main())
