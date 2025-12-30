import asyncio
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class PineconeClient:
    """
    Singleton Pinecone client.
    - Index is created/validated at startup
    - Async-safe upsert/query via executor
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self):
        """
        Initialize Pinecone client and index.
        This MUST be called at application startup.
        """
        if self._initialized:
            return

        logger.info("Initializing Pinecone client")

        self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        existing_indexes = [
            idx["name"] for idx in self._pc.list_indexes()
        ]

        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(
                f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}"
            )

            self._pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION,
                ),
            )

        self._index = self._pc.Index(settings.PINECONE_INDEX_NAME)
        self._initialized = True

        logger.info("Pinecone index is ready")

    # -------------------------
    # Async Operations
    # -------------------------

    async def upsert(
        self,
        vectors: list,
        namespace: str,
        batch_size: int = 100,
    ):
        """
        Async upsert vectors into Pinecone.
        """
        if not self._initialized:
            raise RuntimeError(
                "PineconeClient not initialized. "
                "Call initialize() at startup."
            )

        loop = asyncio.get_running_loop()

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]

            await loop.run_in_executor(
                None,
                lambda b=batch: self._index.upsert(
                    vectors=b,
                    namespace=namespace,
                ),
            )

        logger.info(
            f"Upsert completed | vectors={len(vectors)} | namespace={namespace}"
        )

    async def query(
        self,
        vector: list,
        namespace: str,
        top_k: int = 5,
        include_metadata: bool = True,
    ):
        """
        Async query from Pinecone.
        """
        if not self._initialized:
            raise RuntimeError(
                "PineconeClient not initialized. "
                "Call initialize() at startup."
            )

        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None,
            lambda: self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata,
            ),
        )

        return result
