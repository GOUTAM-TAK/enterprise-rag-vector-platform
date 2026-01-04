import httpx
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class NvidiaLLMClient:
    """
    NVIDIA LLM streaming client.
    """

    def __init__(self, model_name: str | None = None):
        self.model = model_name or settings.NVIDIA_MODEL
        self.base_url = settings.NVIDIA_BASE_URL
        self.api_key = settings.NVIDIA_API_KEY

    async def stream(self, prompt: str):
        """
        Async generator yielding tokens.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    if line.strip() == "data: [DONE]":
                        break

                    try:
                        data = line.removeprefix("data: ").strip()
                        chunk = httpx.Response(200, content=data).json()

                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content")

                        if content:
                            yield content

                    except Exception as e:
                        logger.warning(f"Stream parse error: {e}")
                        continue
