from fastapi import HTTPException
from app.core.config import settings


class RAGUtils:
    @staticmethod
    def validate_rag_access_level(
        rag_access_level: str,
        *,
        raise_http: bool = True,
    ) -> tuple[str, int]:
        """
        Validate RAG access level and return normalized value + rank.

        :param rag_access_level: input access level
        :param raise_http: raise HTTPException (True) or ValueError (False)
        :return: (normalized_level, access_rank)
        """

        if not rag_access_level:
            msg = "rag_access_level is required"
            if raise_http:
                raise HTTPException(status_code=400, detail=msg)
            raise ValueError(msg)

        level = rag_access_level.lower().strip()

        if level not in settings.RAG_ACCESS_LEVELS:
            allowed = ", ".join(settings.RAG_ACCESS_LEVELS.keys())
            msg = f"Invalid rag_access_level '{rag_access_level}'. Allowed values: {allowed}"

            if raise_http:
                raise HTTPException(status_code=400, detail=msg)
            raise ValueError(msg)

        return level, settings.RAG_ACCESS_LEVELS[level]
