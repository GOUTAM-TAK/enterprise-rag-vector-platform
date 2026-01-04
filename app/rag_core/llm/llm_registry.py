from app.rag_core.llm.nvidia_client import NvidiaLLMClient
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class LLMRegistry:
    """
    Holds pre-initialized LLM clients.
    """

    def __init__(self):
        self._models: dict[str, NvidiaLLMClient] = {}

    def initialize(self):
        logger.info("Initializing NVIDIA LLM registry")

        for model_name in settings.nvidia_model_list:
            logger.info(f"Loading NVIDIA model: {model_name}")
            self._models[model_name] = NvidiaLLMClient(model_name)

        logger.info(
            f"NVIDIA LLM registry ready | models={list(self._models.keys())}"
        )

    def get(self, model_name: str | None):
        if not model_name:
            return self._models[settings.NVIDIA_DEFAULT_MODEL]

        if model_name not in self._models:
            raise ValueError(f"Model not available: {model_name}")

        return self._models[model_name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())
